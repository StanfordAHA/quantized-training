import os
import urllib.request
import zipfile

import networkx as nx
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

DATASET_URL = "https://raw.githubusercontent.com/weihua916/powerful-gnns/master/dataset.zip"
COLLAB_TXT = "dataset/COLLAB/COLLAB.txt"
BATCH_SIZE = 64  # powerful-gnns' pass_data_iteratively minibatch
NUM_CALIBRATION = 10  # training graphs used to calibrate per-tensor scales


def _collab_txt(root):
    path = os.path.join(root, COLLAB_TXT)
    if not os.path.exists(path):
        os.makedirs(root, exist_ok=True)
        zip_path = os.path.join(root, "dataset.zip")
        urllib.request.urlretrieve(DATASET_URL, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extract(COLLAB_TXT, root)
    return path


def _load_graphs(path):
    """powerful-gnns' util.load_data(..., degree_as_tag=True), keeping only what GIN reads."""
    graphs, labels, label_dict = [], [], {}
    with open(path) as f:
        for _ in range(int(f.readline().strip())):
            n, l = [int(w) for w in f.readline().strip().split()]
            if l not in label_dict:
                label_dict[l] = len(label_dict)
            g = nx.Graph()
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                row = [int(w) for w in row[:int(row[1]) + 2]]  # [tag, #neighbors, neighbors...]
                for k in row[2:]:
                    g.add_edge(j, k)
            assert len(g) == n
            graphs.append(g)
            labels.append(label_dict[l])

    # Tags are node degrees in g's node *insertion* order (add_edge inserts a neighbor before its own
    # add_node), not node-id order. powerful-gnns uses them as node-id rows all the same; kept as is.
    node_tags = [list(dict(g.degree).values()) for g in graphs]
    tagset = set([])
    for tags in node_tags:
        tagset = tagset.union(set(tags))
    tagset = list(tagset)  # this order sets the one-hot column order
    tag2index = {tagset[i]: i for i in range(len(tagset))}
    return graphs, labels, node_tags, tag2index


def _batch(graphs, labels, node_tags, tag2index, dtype):
    """Dense inputs for one minibatch, built like GraphCNN.forward builds them (learn_eps=False,
    sum neighbor pooling, average graph pooling with custom_ops)."""
    sizes = [len(g) for g in graphs]
    starts = np.cumsum([0] + sizes)
    num_nodes = int(starts[-1])

    x = torch.zeros(num_nodes, len(tag2index))
    for g, tags, start in zip(graphs, node_tags, starts):
        x[torch.arange(len(tags)) + int(start), [tag2index[t] for t in tags]] = 1

    idx = []
    for g, start in zip(graphs, starts):
        edges = [list(pair) for pair in g.edges()]
        edges.extend([[i, j] for j, i in edges])
        idx.append(torch.LongTensor(edges).transpose(0, 1) + int(start))
    idx.append(torch.arange(num_nodes).repeat(2, 1))  # self-loops
    idx = torch.cat(idx, 1)
    # Duplicate indices sum, as in powerful-gnns' sparse Adj_block.to_dense().
    adj = torch.zeros(num_nodes, num_nodes).index_put_(
        (idx[0], idx[1]), torch.ones(idx.shape[1]), accumulate=True
    )

    # 1 / bf16(n) in bf16 equals powerful-gnns' LUT reciprocal (custom_ops.div) bit for bit.
    graph_pool = torch.zeros(len(graphs), num_nodes, dtype=torch.bfloat16)
    for i, (n, start) in enumerate(zip(sizes, starts)):
        graph_pool[i, int(start):int(start) + n] = 1 / torch.tensor(float(n)).bfloat16()

    return {
        "x": x.to(dtype),
        "adj": adj.to(dtype),
        "graph_pool": graph_pool.to(dtype),
        "labels": torch.LongTensor(labels),
    }


def retrieve_dataset(root=os.path.expanduser("~/.cache/powerful-gnns"), dtype=torch.bfloat16):
    """COLLAB fold 0 as dense GIN inputs, matching powerful-gnns (origin/gin) run with
    reproduce_COLLAB.sh: --degree_as_tag, seed 0, fold_idx 0.

    Returns {"test": [batch, ...], "calibration": batch}. "test" holds the fold-0 test graphs in
    fold order, 64 per batch. "calibration" holds the first 10 fold-0 training graphs. Each batch
    is {"x": N×F, "adj": N×N, "graph_pool": G×N, "labels": G}, on the CPU.
    """
    graphs, labels, node_tags, tag2index = _load_graphs(_collab_txt(root))

    # powerful-gnns' separate_data(graphs, seed=0, fold_idx=0)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    train_idx, test_idx = list(skf.split(np.zeros(len(labels)), labels))[0]

    def batch(indices):
        return _batch(
            [graphs[i] for i in indices], [labels[i] for i in indices],
            [node_tags[i] for i in indices], tag2index, dtype,
        )

    return {
        "test": [batch(test_idx[i:i + BATCH_SIZE]) for i in range(0, len(test_idx), BATCH_SIZE)],
        "calibration": batch(train_idx[:NUM_CALIBRATION]),
    }
