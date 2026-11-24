import numpy
import torch
from numpy.core.multiarray import _reconstruct
from torch import nn

from quantized_training import compile, convert_pt2e, prepare_pt2e, transform
from quantized_training.codegen.utils import get_conv_bn_layers

from .utils import get_compile_args, get_transform_args

# Only fc is quantized; every conv, and the residual adds inside layer{1,2,3}, stays bf16.
UNQUANTIZED_MODULES = [r"^conv1$", r"^layer\d"]

# The SiPP checkpoint also pickles numpy state (optimizer, trackers), which torch.load(weights_only=True)
# refuses unless these are allowlisted.
NUMPY_SAFE_GLOBALS = [_reconstruct, numpy.ndarray, numpy.dtype] + [
    type(numpy.dtype(t)) for t in ("f4", "f8", "i8", "i4", "u1", "b1")
]


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        residual = x if self.downsample is None else self.downsample(x)
        out += residual
        return self.relu(out)


def make_layer(inplanes, planes, stride=1):
    downsample = None
    if stride != 1 or inplanes != planes:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes),
        )
    return nn.Sequential(
        BasicBlock(inplanes, planes, stride, downsample),
        BasicBlock(planes, planes),
        BasicBlock(planes, planes),
    )


class ResNet20(nn.Module):
    """cgra_sparse_ml_bench sparse-cnn's CIFAR ResNet(depth=20): BasicBlock, 3 blocks per stage.
    Flattens with torch.flatten (as torchvision's ResNet) instead of x.view(x.size(0), -1)."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = make_layer(16, 16)
        self.layer2 = make_layer(16, 32, stride=2)
        self.layer3 = make_layer(32, 64, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer3(self.layer2(self.layer1(x)))
        return self.fc(torch.flatten(self.avgpool(x), 1))


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(args):
    if args.model_name_or_path is None:
        raise ValueError(
            "spcnn needs --model_name_or_path: the SiPP checkpoint, e.g. cgra_sparse_ml_bench's "
            "sparse-cnn/resnet20_CIFAR10_e182_SiPPNetHybrid_0_p85_rep0_re182_i0"
        )
    torch.serialization.add_safe_globals(NUMPY_SAFE_GLOBALS)
    net = torch.load(args.model_name_or_path, map_location="cpu", weights_only=True)["net"]
    # Same key mapping as sparse-cnn/main.py, but strict: main.py's strict=False would drop a
    # mismatched weight silently.
    state_dict = {
        k.replace("compressed_net.", "").replace("torchnet.", ""): v
        for k, v in net.items() if "torchnet" in k
    }
    model = ResNet20()
    model.load_state_dict(state_dict, strict=True)
    model = model.eval().to(torch.bfloat16 if args.bf16 else torch.float32)
    # Fold BN after the dtype cast, as sparse-cnn/test.py and the torchvision ResNet path do.
    return torch.ao.quantization.fuse_modules(model, get_conv_bn_layers(model), inplace=True)


def quantize_and_dump_model(model, quantizer, dataset, vector_stages, args):
    device = get_device()
    transform_args = get_transform_args(args, vector_stages)
    compile_args = get_compile_args(args)

    for name in UNQUANTIZED_MODULES:
        quantizer.set_module_name(name, None)

    example_args = (dataset["test"][0][0][None].to(device),)
    gm = prepare_pt2e(model.to(device), quantizer, example_args)

    # Calibrate per-tensor scales one training image per call, like the torchvision ResNet path.
    # The amax observer sets each call's scale from earlier calls only, so the frozen scale comes
    # from images 1..9 of the 10. Microscaling scales ignore calibration.
    with torch.no_grad():
        for image in dataset["calibration"]:
            gm(image[None].to(device))

    convert_pt2e(gm, args.bias)

    old_output = gm(*example_args)

    transform(gm, example_args, **transform_args)
    gm.graph.print_tabular()

    new_output = gm(*example_args)

    compile(gm, example_args, **compile_args)
    return gm, old_output, new_output


def evaluate(model, dataset):
    """Prints and returns the number of correctly classified test images, run one at a time."""
    device = get_device()
    model.to(device)
    correct = 0
    with torch.no_grad():
        for image, label in dataset["test"]:
            correct += int(model(image[None].to(device)).argmax(dim=-1).item() == label)
    total = len(dataset["test"])
    print(f"{model.__class__.__name__} Accuracy: {correct}/{total} = {correct / total:.4f}")
    return correct
