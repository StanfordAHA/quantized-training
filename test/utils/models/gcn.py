import torch
import torch.nn.functional as F
from torch import nn

from quantized_training import compile, convert_pt2e, prepare_pt2e, transform

from .utils import get_compile_args, get_transform_args

# Only linear2 (the last, fully dense GEMM) is quantized; these stay bf16.
UNQUANTIZED_MODULES = ["^normalize$", "^aggr1$", "^linear1$", "^aggr2$"]


class Normalize(nn.Module):
    """Â = D^-1/2 @ A @ D^-1/2."""

    def forward(self, a, d):
        return torch.matmul(torch.matmul(d, a), d)


class Aggregate(nn.Module):
    def forward(self, adj, h):
        return torch.matmul(adj, h)


class Transform(nn.Module):
    """h @ weight, with weight stored (in, out) and no bias, as in the ssp GCN."""

    def __init__(self, weight):
        super().__init__()
        self.weight = nn.Parameter(weight)

    def forward(self, h):
        return torch.matmul(h, self.weight)


class GCN(nn.Module):
    """Dense two-layer GCN from cgra_sparse_ml_bench's ssp (CiteSeer, no self-loops)."""

    def __init__(self, w1, w2):
        super().__init__()
        self.normalize = Normalize()
        self.aggr1 = Aggregate()
        self.linear1 = Transform(w1)
        self.aggr2 = Aggregate()
        self.linear2 = Transform(w2)

    def forward(self, a, d, x):
        adj = self.normalize(a, d)
        # F.relu, not torch.relu: codegen only fuses F.relu / nn.ReLU into the GEMM before it.
        h = F.relu(self.linear1(self.aggr1(adj, x)))
        return self.linear2(self.aggr2(adj, h))


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(args):
    if args.model_name_or_path is None:
        raise ValueError(
            "gcn needs --model_name_or_path: the .pt written by cgra_sparse_ml_bench's "
            "`quantized_training_parity.py export-weights`"
        )
    weights = torch.load(args.model_name_or_path, map_location="cpu", weights_only=True)
    model = GCN(weights["w1"], weights["w2"]).eval()
    return model.to(torch.bfloat16 if args.bf16 else torch.float32)


def _inputs(dataset, device):
    return tuple(dataset[k].to(device) for k in ("a", "d", "x"))


def quantize_and_dump_model(model, quantizer, dataset, vector_stages, args):
    # Runs on the GPU when available: CPU bf16 GEMMs drift from ssp's GPU-computed dump.
    device = get_device()
    transform_args = get_transform_args(args, vector_stages)
    compile_args = get_compile_args(args)

    for name in UNQUANTIZED_MODULES:
        quantizer.set_module_name(name, None)

    example_args = _inputs(dataset, device)
    gm = prepare_pt2e(model.to(device), quantizer, example_args)

    # Calibrate per-tensor scales on the whole graph, which is exactly what inference quantizes.
    # Two passes: the amax observer sets each pass's scale from earlier passes only, so one pass
    # leaves scale = 1.0. Microscaling scales ignore calibration.
    with torch.no_grad():
        for _ in range(2):
            gm(*example_args)

    convert_pt2e(gm, args.bias)

    old_output = gm(*example_args)

    transform(gm, example_args, **transform_args)
    gm.graph.print_tabular()

    new_output = gm(*example_args)

    compile(gm, example_args, **compile_args)
    return gm, old_output, new_output


def evaluate(model, dataset):
    """Prints and returns the number of correctly classified test nodes."""
    device = get_device()
    model.to(device)
    with torch.no_grad():
        logits = model(*_inputs(dataset, device))
    mask = dataset["test_mask"].to(device)
    pred = logits[mask].float().argmax(dim=-1)
    correct = int((pred == dataset["labels"].to(device)[mask]).sum())
    total = int(mask.sum())
    print(f"{model.__class__.__name__} Accuracy: {correct}/{total} = {correct / total:.4f}")
    return correct
