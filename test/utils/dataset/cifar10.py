import os

import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10

NUM_CALIBRATION = 10  # training images used to calibrate per-tensor scales
MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2023, 0.1994, 0.2010]


def retrieve_dataset(root=os.path.expanduser("~/.cache/cifar10"), dtype=torch.bfloat16):
    """CIFAR-10 with cgra_sparse_ml_bench sparse-cnn's test transform (data.py): ToTensor and
    Normalize in fp32, then cast to `dtype`.

    Returns {"test": the 10k-image test set, items (3x32x32 image, label),
    "calibration": the first NUM_CALIBRATION training images, NUM_CALIBRATION x 3 x 32 x 32}.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.ConvertImageDtype(dtype),
    ])
    test = CIFAR10(root, train=False, download=True, transform=transform)
    train = CIFAR10(root, train=True, download=True, transform=transform)
    calibration = torch.stack([train[i][0] for i in range(NUM_CALIBRATION)])
    return {"test": test, "calibration": calibration}
