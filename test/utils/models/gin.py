import torch
import torch.nn.functional as F
from torch import nn

from quantized_training import compile, convert_pt2e, prepare_pt2e, transform

from .utils import get_compile_args, get_transform_args

# Only the prediction heads' linears (heads.{l}.linear) are quantized; these stay bf16.
# The score sum is listed because per-tensor's global config would otherwise quantize an add of
# two activations.
UNQUANTIZED_MODULES = [r"^layers\.", r"^heads\.\d+\.pool$", r"^score_add$"]


class Aggregate(nn.Module):
    """adj @ h (neighbor sum) or graph_pool @ h (graph average)."""

    def forward(self, adj, h):
        return torch.matmul(adj, h)


class Transform(nn.Module):
    """h @ weight, with weight stored (in, out) as powerful-gnns dumps it (C.npy = Wᵀ)."""

    def __init__(self, weight):
        super().__init__()
        self.weight = nn.Parameter(weight)

    def forward(self, h):
        return torch.matmul(h, self.weight)


class Add(nn.Module):
    def forward(self, a, b):
        return a + b


class GINLayer(nn.Module):
    """GINConv with a 2-layer MLP whose BatchNorms are folded into its linears."""

    def __init__(self, w0, b0, w1, b1):
        super().__init__()
        self.aggr = Aggregate()
        self.linear0 = Transform(w0)
        self.b0 = nn.Parameter(b0)
        self.linear1 = Transform(w1)
        self.b1 = nn.Parameter(b1)

    def forward(self, adj, h):
        # Bias adds are separate bf16 ops, as in powerful-gnns' dump.
        # F.relu, not torch.relu: codegen only fuses F.relu / nn.ReLU into the GEMM before it.
        h = F.relu(self.linear0(self.aggr(adj, h)) + self.b0)
        return F.relu(self.linear1(h) + self.b1)


class Head(nn.Module):
    """One prediction layer: graph pooling, then a linear to class scores."""

    def __init__(self, w, b):
        super().__init__()
        self.pool = Aggregate()
        self.linear = Transform(w)
        self.b = nn.Parameter(b)

    def forward(self, graph_pool, h):
        return self.linear(self.pool(graph_pool, h)) + self.b


class GIN(nn.Module):
    """Dense GIN from cgra_sparse_ml_bench's powerful-gnns (origin/gin), as it runs in eval mode
    after fuse_linear_bn: sum neighbor pooling, learn_eps=False, one head per hidden
    representation (input included), head scores summed."""

    def __init__(self, weights):
        super().__init__()
        num_layers = len({k.split(".")[1] for k in weights if k.startswith("layers.")})
        num_heads = len({k.split(".")[1] for k in weights if k.startswith("heads.")})
        self.layers = nn.ModuleList(
            GINLayer(*(weights[f"layers.{l}.{n}"] for n in ("w0", "b0", "w1", "b1")))
            for l in range(num_layers)
        )
        self.heads = nn.ModuleList(
            Head(weights[f"heads.{l}.w"], weights[f"heads.{l}.b"]) for l in range(num_heads)
        )
        self.score_add = Add()

    def forward(self, x, adj, graph_pool):
        hidden = [x]
        for layer in self.layers:
            hidden.append(layer(adj, hidden[-1]))
        score = self.heads[0](graph_pool, hidden[0])
        # Index, don't slice: export names a sliced ModuleList's children
        # "heads.slice(1, None, None)._modules.0", which UNQUANTIZED_MODULES would not match.
        for l in range(1, len(self.heads)):
            score = self.score_add(score, self.heads[l](graph_pool, hidden[l]))
        return score


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(args):
    if args.model_name_or_path is None:
        raise ValueError(
            "gin needs --model_name_or_path: the .pt written by cgra_sparse_ml_bench's "
            "`quantized_training_parity.py export-weights --model gin`"
        )
    weights = torch.load(args.model_name_or_path, map_location="cpu", weights_only=True)
    model = GIN(weights).eval()
    return model.to(torch.bfloat16 if args.bf16 else torch.float32)


def _inputs(batch, device):
    return tuple(batch[k].to(device) for k in ("x", "adj", "graph_pool"))


def quantize_and_dump_model(model, quantizer, dataset, vector_stages, args):
    # Runs on the GPU when available: CPU bf16 GEMMs drift from powerful-gnns' GPU-computed dump.
    device = get_device()
    transform_args = get_transform_args(args, vector_stages)
    compile_args = get_compile_args(args)

    for name in UNQUANTIZED_MODULES:
        quantizer.set_module_name(name, None)

    # The first test batch is the one powerful-gnns dumps.
    example_args = _inputs(dataset["test"][0], device)
    gm = prepare_pt2e(model.to(device), quantizer, example_args)

    # Calibrate per-tensor scales on 10 training graphs (COLLAB has no calibration split). Two
    # passes: the amax observer sets each pass's scale from earlier passes only, so one pass
    # leaves scale = 1.0. Microscaling scales ignore calibration.
    calibration_args = _inputs(dataset["calibration"], device)
    with torch.no_grad():
        for _ in range(2):
            gm(*calibration_args)

    convert_pt2e(gm, args.bias)

    old_output = gm(*example_args)

    transform(gm, example_args, **transform_args)
    gm.graph.print_tabular()

    new_output = gm(*example_args)

    compile(gm, example_args, **compile_args)
    return gm, old_output, new_output


def evaluate(model, dataset):
    """Prints and returns the number of correctly classified test graphs."""
    device = get_device()
    model.to(device)
    correct = total = 0
    with torch.no_grad():
        for batch in dataset["test"]:
            pred = model(*_inputs(batch, device)).float().argmax(dim=-1)
            correct += int((pred == batch["labels"].to(device)).sum())
            total += len(batch["labels"])
    print(f"{model.__class__.__name__} Accuracy: {correct}/{total} = {correct / total:.4f}")
    return correct
