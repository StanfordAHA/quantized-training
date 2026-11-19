import os

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid


def retrieve_dataset(root=os.path.expanduser("~/.cache/planetoid"), device=None):
    """CiteSeer as dense bf16 GCN inputs: adjacency A, diagonal D^-1/2 and features X.

    Built the way cgra_sparse_ml_bench's ssp GCN builds them in bf16 (no self-loops,
    bf16 degrees and pow(-0.5), isolated nodes -> 0), with ssp's "complete" split:
    the last 500 nodes are the test set. `device` is only used to compute D^-1/2;
    every returned tensor is on the CPU.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    data = Planetoid(root, "CiteSeer", transform=T.NormalizeFeatures())[0]
    num_nodes = data.num_nodes
    row, col = data.edge_index

    a = torch.zeros(num_nodes, num_nodes, dtype=torch.bfloat16)
    a[row, col] = 1

    d_inv_sqrt = a.to(device).sum(dim=0).pow(-0.5)
    d_inv_sqrt.masked_fill_(d_inv_sqrt == float("inf"), 0)

    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[num_nodes - 500:] = True

    return {
        "a": a,
        "d": torch.diag(d_inv_sqrt).cpu(),
        "x": data.x.bfloat16(),
        "labels": data.y,
        "test_mask": test_mask,
    }
