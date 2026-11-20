"""Load a state-spaces Mamba2 checkpoint and adapt it for Voyager's compiler.

``load_model`` builds transformers' ``Mamba2ForCausalLM`` from a checkpoint in
the ``mamba_ssm`` format (for example ``state-spaces/mamba2-1.3b``).
``Mamba2Prefill`` runs one prefill over that model's modules and weights, with
each Mamba2 mixer written out as PyTorch operators. Its SSD scan follows
``mamba_ssm``'s chunked algorithm (``ssd_minimal_discrete``): per chunk, the
Ct@B, score@x, B@X and C@states GEMMs, and a sequential state pass between
chunks that starts from a zero state.
"""

import json
import os

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch import nn

MAMBA2_MODEL_ID = "state-spaces/mamba2-1.3b"
MAMBA2_REVISION = "c5b59d00ec85d313adea86a08cad2a43c962dd3b"
MAMBA2_TOKENIZER_ID = "EleutherAI/gpt-neox-20b"

# Defaults of mamba_ssm.modules.mamba2.Mamba2 for the ssm_cfg keys a checkpoint
# may leave out.
MAMBA2_SSM_DEFAULTS = {
    "d_state": 128,
    "d_conv": 4,
    "expand": 2,
    "headdim": 64,
    "ngroups": 1,
    "chunk_size": 256,
}


def load_model(args):
    """Return (model, tokenizer) for a pure-Mamba2 ``mamba_ssm`` checkpoint.

    ``args.model_name_or_path`` is a Hub ID or a local directory holding
    ``config.json`` and ``pytorch_model.bin``; the default checkpoint is pinned
    to ``MAMBA2_REVISION``. The model is in BF16 when ``args.bf16`` is set and
    in FP32 otherwise, and in eval mode.
    """
    from transformers import AutoTokenizer, Mamba2Config, Mamba2ForCausalLM

    if args.model_name_or_path is None:
        args.model_name_or_path = MAMBA2_MODEL_ID
    revision = MAMBA2_REVISION if args.model_name_or_path == MAMBA2_MODEL_ID else None

    def checkpoint_file(filename):
        if os.path.isdir(args.model_name_or_path):
            return os.path.join(args.model_name_or_path, filename)
        return hf_hub_download(args.model_name_or_path, filename, revision=revision)

    with open(checkpoint_file("config.json")) as f:
        config = json.load(f)

    ssm_cfg = dict(config.get("ssm_cfg", {}))
    if (
        ssm_cfg.pop("layer", None) != "Mamba2"
        or config.get("attn_layer_idx")
        or config.get("d_intermediate", 0)
        or not config.get("rms_norm", True)
    ):
        raise ValueError(
            "[ERROR] Expected a checkpoint of Mamba2 layers only, with RMSNorm "
            "and no attention or MLP layers."
        )
    unsupported = set(ssm_cfg) - set(MAMBA2_SSM_DEFAULTS)
    if unsupported:
        raise ValueError(f"[ERROR] Unsupported Mamba2 ssm_cfg keys: {sorted(unsupported)}")
    ssm = {**MAMBA2_SSM_DEFAULTS, **ssm_cfg}

    state_dict = torch.load(
        checkpoint_file("pytorch_model.bin"), map_location="cpu", weights_only=True
    )
    state_dict = {
        name.replace("backbone.embedding.", "backbone.embeddings.", 1): tensor
        for name, tensor in state_dict.items()
    }

    d_model = config["d_model"]
    text_config = Mamba2Config(
        hidden_size=d_model,
        num_hidden_layers=config["n_layer"],
        vocab_size=state_dict["backbone.embeddings.weight"].shape[0],
        state_size=ssm["d_state"],
        conv_kernel=ssm["d_conv"],
        expand=ssm["expand"],
        head_dim=ssm["headdim"],
        num_heads=ssm["expand"] * d_model // ssm["headdim"],
        n_groups=ssm["ngroups"],
        chunk_size=ssm["chunk_size"],
        residual_in_fp32=config.get("residual_in_fp32", True),
        tie_word_embeddings=config.get("tie_embeddings", True),
        rms_norm=True,
        layer_norm_epsilon=1e-5,
        hidden_act="silu",
        time_step_limit=(0.0, float("inf")),
        use_bias=False,
        use_conv_bias=True,
    )

    # Meta initialization avoids allocating a second copy of the weights.
    with torch.device("meta"):
        model = Mamba2ForCausalLM(text_config)

    expected_keys = set(model.state_dict())
    unexpected_keys = set(state_dict) - expected_keys
    missing_keys = expected_keys - set(state_dict)
    if text_config.tie_word_embeddings:
        missing_keys.discard("lm_head.weight")
    if unexpected_keys or missing_keys:
        raise ValueError(
            f"[ERROR] Mamba2 checkpoint mismatch: unexpected {sorted(unexpected_keys)}, "
            f"missing {sorted(missing_keys)}"
        )

    model.load_state_dict(state_dict, strict=False, assign=True)
    model.tie_weights()
    model.to(torch.bfloat16 if args.bf16 else torch.float32).eval()

    tokenizer = AutoTokenizer.from_pretrained(MAMBA2_TOKENIZER_ID)
    return model, tokenizer


def causal_depthwise_conv1d_silu(x, weight, bias, zero_prefix):
    """Return SiLU of the Mamba2 block's causal depthwise conv1d, channels-last.

    ``x`` is ``[batch, length, channels]``, ``weight`` is ``[channels, width]``,
    ``bias`` is ``[channels]`` or None, and ``zero_prefix`` is
    ``[batch, width - 1, channels]`` zeros. Each output position of a channel is
    that channel's filter applied to the current and the previous width-1
    positions, with zeros before position 0, plus the bias:
    ``out[l] = bias + sum_k weight[:, k] * x[l - width + 1 + k]``. It is written
    as shifted multiply-adds on the channels-last layout, so the graph has no
    channels-first transposes and no conv1d op.
    """
    length = x.shape[1]
    width = weight.shape[1]
    padded = torch.cat([zero_prefix, x], dim=1)                  # [N, length + width - 1, C]
    out = padded[:, :length] * weight[:, 0]
    for k in range(1, width):
        out = out + padded[:, k : k + length] * weight[:, k]
    if bias is not None:
        out = out + bias
    return F.silu(out)


class Mamba2Scan(nn.Module):
    """Chunked SSD scan of one Mamba2 layer, starting from a zero state.

    Shapes: ``N`` batch (1), ``L`` positions, ``Q`` positions per chunk,
    ``K = L / Q`` chunks, ``H`` heads of ``P`` channels, ``G`` groups of
    ``Hg = H / G`` heads, ``S`` state size. The heads of a group share B and C.

    Per chunk, with ``X = x * dt`` and ``a = A * dt``:
      1. C@B^T, per group:  scores = C @ B^T                         [Q, Q]
      2. score@x, per head: Y_diag = (scores * decay) @ X             [Q, P]
      3. B@X, per group:    state^T = B^T @ (X * decay_to_end), with the
                            group's heads side by side as columns     [S, Hg*P]
      4. State passing:     entering^T[k+1] = exp(sum of a over chunk k)
                            * entering^T[k] + state^T[k], entering^T[0] = 0
      5. C@states, per group: Y_off = exp(a_cum) * (C @ entering^T)   [Q, Hg*P]
    ``decay[l, s] = exp(a[s+1] + ... + a[l])`` for ``l >= s`` and 0 above the
    diagonal. ``decay_to_end`` is its last row and ``a_cum`` the running sum
    of ``a`` within the chunk. Steps 1 and 3 read separate copies of B^T,
    because step 1 MX-quantizes it along the states and step 3 along the
    positions. The states stay transposed from step 3 to step 5, so neither X
    nor the states are transposed for a GEMM.
    """

    def __init__(self, mixer, chunk_size):
        super().__init__()
        self.num_heads = mixer.num_heads
        self.head_dim = mixer.head_dim
        self.n_groups = mixer.n_groups
        self.state_size = mixer.ssm_state_size
        self.chunk_size = chunk_size

        dtype = mixer.A_log.dtype
        group_heads = self.num_heads // self.n_groups
        self.register_buffer(
            "decay_rate", -mixer.A_log.detach().float().exp().to(dtype), persistent=False
        )
        self.register_buffer(
            "strict_lower_mask",
            torch.ones(chunk_size, chunk_size, dtype=dtype).tril(-1),
            persistent=False,
        )
        self.register_buffer(
            "lower_mask",
            torch.ones(chunk_size, chunk_size, dtype=dtype).tril(0),
            persistent=False,
        )
        self.register_buffer(
            "initial_state",
            torch.zeros(
                1, self.n_groups, self.state_size, group_heads, self.head_dim, dtype=dtype
            ),
            persistent=False,
        )

    def forward(self, x, dt, B, C):
        """Return the scan output ``[N, L, H, P]``, without the D skip term.

        ``x`` is ``[N, L, H, P]``, ``dt`` is ``[N, L, H]`` after softplus, and
        ``B`` and ``C`` are ``[N, L, G, S]``.
        """
        batch, length = x.shape[:2]
        Q, H, P = self.chunk_size, self.num_heads, self.head_dim
        G, S = self.n_groups, self.state_size
        Hg = H // G
        if batch != 1:
            raise ValueError(f"[ERROR] Mamba2 scan supports batch size 1, got {batch}.")
        if length % Q:
            raise ValueError(
                f"[ERROR] Mamba2 scan needs a prefill length divisible by {Q}, got {length}."
            )
        K = length // Q

        def group_major(t):
            """[N, K, Q, G, F] -> [N, K, G, Q, F]; a reshape when there is one group."""
            return t.reshape(batch, K, 1, Q, -1) if G == 1 else t.permute(0, 1, 3, 2, 4)

        # Chunk-major layouts; a and X are head-major for the per-head score@x.
        a = (dt * self.decay_rate).reshape(batch, K, Q, G, Hg).permute(0, 1, 3, 4, 2)
        a_cum = torch.cumsum(a, dim=-1)                                   # [N, K, G, Hg, Q]
        xdt = x * dt[..., None]                                           # [N, L, H, P]
        X = xdt.reshape(batch, K, Q, G, Hg, P).permute(0, 1, 3, 4, 2, 5)  # [N, K, G, Hg, Q, P]
        # Two separate B^T tensors: the GEMMs quantize B^T along different axes (the
        # states for C@B^T, the positions for B^T@X_end), so each needs its own.
        B_t_scores = B.reshape(batch, K, Q, G, S).permute(0, 1, 3, 4, 2)  # [N, K, G, S, Q]
        B_t_states = B.reshape(batch, K, Q, G, S).permute(0, 1, 3, 4, 2)  # [N, K, G, S, Q]
        C = group_major(C.reshape(batch, K, Q, G, S))                     # [N, K, G, Q, S]

        # decay[l, s]: row l is the output position, column s the source.
        segment_sum = torch.cumsum(a[..., :, None] * self.strict_lower_mask, dim=-2)
        decay = torch.exp(segment_sum) * self.lower_mask                 # [N, K, G, Hg, Q, Q]

        # 1. C@B^T and 2. score@x
        scores = torch.matmul(C, B_t_scores)                              # [N, K, G, Q, Q]
        y_diag = torch.matmul(scores[:, :, :, None] * decay, X)           # [N, K, G, Hg, Q, P]

        # 3. B@X as B^T @ X_end, with X_end position-major [Q, Hg*P]
        decay_to_end = decay[..., -1, :].transpose(-1, -2)               # [N, K, G, Q, Hg]
        X_end = group_major(xdt.reshape(batch, K, Q, G, Hg * P)).reshape(batch, K, G, Q, Hg, P)
        X_end = (X_end * decay_to_end[..., None]).reshape(batch, K, G, Q, Hg * P)
        chunk_states_t = torch.matmul(B_t_states, X_end).reshape(batch, K, G, S, Hg, P)

        # 4. State passing, on the transposed states
        chunk_decay = torch.exp(a_cum[..., -1])                           # [N, K, G, Hg]
        entering = [self.initial_state]
        for k in range(1, K):
            entering.append(
                entering[-1] * chunk_decay[:, k - 1, :, None, :, None]
                + chunk_states_t[:, k - 1]
            )
        entering = torch.stack(entering, dim=1).reshape(batch, K, G, S, Hg * P)

        # 5. C@states
        y_off = torch.matmul(C, entering).reshape(batch, K, G, Q, Hg, P)
        y_off = y_off.permute(0, 1, 2, 4, 3, 5) * torch.exp(a_cum)[..., None]

        y = (y_diag + y_off).permute(0, 1, 4, 2, 3, 5)                    # [N, K, Q, G, Hg, P]
        return y.reshape(batch, length, H, P)


class Mamba2Prefill(nn.Module):
    """A cache-free prefill over a transformers ``Mamba2ForCausalLM``'s modules.

    Registers the model's backbone and head without copying weights, one
    ``Mamba2Scan`` per layer, and per layer three linears that split
    ``in_proj`` by output rows: ``z_proj`` (the gate z), ``xBC_proj`` (the
    conv input xBC) and ``dt_proj`` (dt). Their weights are copies of the
    matching ``in_proj`` rows, so the results equal ``in_proj``'s; ``in_proj``
    itself is left in place and unused.
    ``forward`` takes ``[N, L, hidden]`` embeddings and returns ``[N, L, vocab]``
    logits. Every block runs its RMSNorm, the mixer written out by
    ``mixer_forward`` and the residual add, all in the dtype of the embeddings;
    then ``norm_f`` and ``lm_head`` run.
    """

    def __init__(self, model, chunk_size=None):
        super().__init__()
        self.backbone = model.backbone
        self.lm_head = model.lm_head
        self.config = model.config
        chunk_size = chunk_size or model.config.chunk_size

        scans = []
        for block in self.backbone.layers:
            mixer = block.mixer
            limits = tuple(float(v) for v in mixer.time_step_limit)
            if limits != (0.0, float("inf")):
                raise ValueError(
                    "[ERROR] Mamba2 export leaves out the dt clamp; "
                    f"got time_step_limit {limits}."
                )
            if mixer.activation not in ("silu", "swish"):
                raise ValueError(
                    "[ERROR] Mamba2 export applies SiLU after the convolution; "
                    f"got activation {mixer.activation!r}."
                )
            scans.append(Mamba2Scan(mixer, chunk_size))
        self.scans = nn.ModuleList(scans)

        # in_proj split by output rows: z, then xBC, then dt.
        z_projs, xBC_projs, dt_projs = [], [], []
        for block in self.backbone.layers:
            mixer = block.mixer
            if mixer.in_proj.bias is not None:
                raise ValueError("[ERROR] Mamba2 export splits in_proj by rows and expects no bias.")
            weight = mixer.in_proj.weight
            hidden, rows = weight.shape[1], weight.shape[0]
            z_end = mixer.intermediate_size
            xBC_end = z_end + mixer.conv_dim
            if rows != xBC_end + mixer.num_heads:
                raise ValueError(
                    f"[ERROR] Mamba2 export expects in_proj rows z, xBC, dt = {xBC_end + mixer.num_heads}, "
                    f"got {rows}."
                )
            factory = dict(bias=False, device=weight.device, dtype=weight.dtype)
            z_proj = nn.Linear(hidden, z_end, **factory)
            xBC_proj = nn.Linear(hidden, mixer.conv_dim, **factory)
            dt_proj = nn.Linear(hidden, mixer.num_heads, **factory)
            with torch.no_grad():
                z_proj.weight.copy_(weight[:z_end])
                xBC_proj.weight.copy_(weight[z_end:xBC_end])
                dt_proj.weight.copy_(weight[xBC_end:])
            z_projs.append(z_proj)
            xBC_projs.append(xBC_proj)
            dt_projs.append(dt_proj)
        self.z_projs = nn.ModuleList(z_projs)
        self.xBC_projs = nn.ModuleList(xBC_projs)
        self.dt_projs = nn.ModuleList(dt_projs)

        # Zeros before position 0 of the causal conv, shared by all layers (batch size 1).
        mixer = self.backbone.layers[0].mixer
        self.register_buffer(
            "conv_zero_prefix",
            torch.zeros(1, mixer.conv_kernel_size - 1, mixer.conv_dim, dtype=mixer.A_log.dtype),
            persistent=False,
        )

    def mixer_forward(self, mixer, scan, z_proj, xBC_proj, dt_proj, hidden_states):
        """Return one Mamba2 mixer's output for ``[N, L, hidden]`` inputs.

        Steps: the gate silu(z) from z_proj; xBC from xBC_proj and dt from
        dt_proj; causal conv1d + SiLU on xBC; split into x, B, C; dt =
        softplus(dt + dt_bias); the chunked scan; the D skip term; the gated
        grouped RMSNorm norm(y * silu(z)); out_proj.
        """
        batch, length, _ = hidden_states.shape
        groups_states = mixer.n_groups * mixer.ssm_state_size

        gate = F.silu(z_proj(hidden_states))
        xBC = xBC_proj(hidden_states)
        dt = dt_proj(hidden_states)
        xBC = causal_depthwise_conv1d_silu(
            xBC, mixer.conv1d.weight.squeeze(1), mixer.conv1d.bias, self.conv_zero_prefix
        )
        x, B, C = xBC.split([mixer.intermediate_size, groups_states, groups_states], dim=-1)
        dt = F.softplus(dt + mixer.dt_bias)

        x = x.reshape(batch, length, mixer.num_heads, mixer.head_dim)
        y = scan(
            x,
            dt,
            B.reshape(batch, length, mixer.n_groups, mixer.ssm_state_size),
            C.reshape(batch, length, mixer.n_groups, mixer.ssm_state_size),
        )
        y = (y + x * mixer.D[:, None]).reshape(batch, length, mixer.intermediate_size)

        # Gated RMSNorm with the gate applied first, normalizing each group of
        # intermediate_size / n_groups channels.
        group_size = mixer.intermediate_size // mixer.n_groups
        gated = (y * gate).reshape(batch, length, mixer.n_groups, group_size)
        rstd = torch.rsqrt(gated.square().mean(-1, keepdim=True) + mixer.norm.variance_epsilon)
        y = (gated * rstd).reshape(batch, length, mixer.intermediate_size) * mixer.norm.weight
        return mixer.out_proj(y)

    def forward(self, inputs_embeds):
        hidden_states = inputs_embeds
        layers = zip(self.backbone.layers, self.scans, self.z_projs, self.xBC_projs, self.dt_projs)
        for block, scan, z_proj, xBC_proj, dt_proj in layers:
            residual = hidden_states
            hidden_states = self.mixer_forward(
                block.mixer, scan, z_proj, xBC_proj, dt_proj, block.norm(hidden_states)
            )
            hidden_states = residual + hidden_states
        hidden_states = self.backbone.norm_f(hidden_states)
        return self.lm_head(hidden_states)
