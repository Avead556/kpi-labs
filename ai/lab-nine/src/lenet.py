import pickle
import random
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

DATA_ROOT = Path(__file__).resolve().parents[2] / ".data"
CIFAR_ARCHIVE = DATA_ROOT / "cifar-10-python.tar.gz"
MNIST_CLASSES = [str(digit) for digit in range(10)]
CIFAR_CLASSES = ["літак", "авто", "птах", "кіт", "олень",
                 "собака", "жаба", "кінь", "корабель", "вантажівка"]

BATCH_SIZE = 100
LEARNING_RATE = 1.0e-3
EVAL_CHUNK = 2000


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def best_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_mnist():
    import torchvision.datasets

    train = torchvision.datasets.MNIST(str(DATA_ROOT), train=True, download=False)
    test = torchvision.datasets.MNIST(str(DATA_ROOT), train=False, download=False)
    return (train.data.unsqueeze(1).float(), train.targets,
            test.data.unsqueeze(1).float(), test.targets)


def load_cifar10():
    batches = {}
    with tarfile.open(CIFAR_ARCHIVE, "r:gz") as archive:
        for member in archive.getmembers():
            name = Path(member.name).name
            if name.startswith(("data_batch_", "test_batch")):
                batches[name] = pickle.load(archive.extractfile(member), encoding="bytes")

    def stack(names):
        images = np.concatenate([batches[name][b"data"] for name in names])
        labels = np.concatenate([batches[name][b"labels"] for name in names])
        images = torch.from_numpy(images.reshape(-1, 3, 32, 32)).float()
        return images, torch.from_numpy(labels).long()

    X_train, y_train = stack([f"data_batch_{index}" for index in range(1, 6)])
    X_test, y_test = stack(["test_batch"])
    return X_train, y_train, X_test, y_test


class LeNet5(nn.Module):

    def __init__(self, in_channels=1, padding=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, padding=padding)
        self.act1 = nn.Tanh()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=0)
        self.act2 = nn.Tanh()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(5 * 5 * 16, 120)
        self.act3 = nn.Tanh()
        self.fc2 = nn.Linear(120, 84)
        self.act4 = nn.Tanh()
        self.fc3 = nn.Linear(84, 10)

    def features(self, x):
        maps = []
        x = self.act1(self.conv1(x))
        maps.append(("conv1 + tanh", x))
        x = self.pool1(x)
        maps.append(("pool1", x))
        x = self.act2(self.conv2(x))
        maps.append(("conv2 + tanh", x))
        x = self.pool2(x)
        maps.append(("pool2", x))
        return maps, x

    def forward(self, x):
        x = self.pool2(self.act2(self.conv2(self.pool1(self.act1(self.conv1(x))))))
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        x = self.act3(self.fc1(x))
        x = self.act4(self.fc2(x))
        return self.fc3(x)


class ConvBlock(nn.Sequential):

    def __init__(self, in_channels, out_channels):
        super().__init__(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU())


class LeNet5Plus(nn.Module):

    def __init__(self, in_channels=1, width=32):
        super().__init__()
        self.stage1 = nn.Sequential(ConvBlock(in_channels, width), ConvBlock(width, width))
        self.pool1 = nn.MaxPool2d(2)
        self.stage2 = nn.Sequential(ConvBlock(width, 2 * width), ConvBlock(2 * width, 2 * width))
        self.pool2 = nn.MaxPool2d(2)
        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Linear(2 * width * 7 * 7, 256),
                                        nn.ReLU(),
                                        nn.Dropout(0.3),
                                        nn.Linear(256, 10))

    def features(self, x):
        maps = []
        x = self.stage1[0](x)
        maps.append(("conv1a 3×3", x))
        x = self.stage1[1](x)
        maps.append(("conv1b 3×3", x))
        x = self.pool1(x)
        maps.append(("maxpool1", x))
        x = self.stage2[0](x)
        maps.append(("conv2a 3×3", x))
        x = self.stage2[1](x)
        maps.append(("conv2b 3×3", x))
        x = self.pool2(x)
        maps.append(("maxpool2", x))
        return maps, x

    def forward(self, x):
        x = self.pool2(self.stage2(self.pool1(self.stage1(x))))
        return self.classifier(x)


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


@torch.no_grad()
def evaluate(model, X, y, device, loss_fn=None):
    loss_fn = loss_fn or nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    total_loss, correct = 0.0, 0
    for start in range(0, len(X), EVAL_CHUNK):
        batch = X[start:start + EVAL_CHUNK].to(device)
        target = y[start:start + EVAL_CHUNK].to(device)
        logits = model(batch)
        total_loss += loss_fn(logits, target).item()
        correct += (logits.argmax(dim=1) == target).sum().item()
    return total_loss / len(X), correct / len(X)


def train(model, data, epochs, device, seed=0, log=print):
    X_train, y_train, X_test, y_test = data
    seed_everything(seed)
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = {key: [] for key in ("train_loss", "train_acc", "test_loss", "test_acc", "seconds")}

    for epoch in range(epochs):
        model.train()
        started = time.perf_counter()
        order = np.random.permutation(len(X_train))
        running_loss = running_correct = 0.0
        for start in range(0, len(X_train), BATCH_SIZE):
            indexes = order[start:start + BATCH_SIZE]
            X_batch = X_train[indexes].to(device)
            y_batch = y_train[indexes].to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss_value = loss_fn(preds, y_batch)
            loss_value.backward()
            optimizer.step()

            running_loss += loss_value.detach() * len(indexes)
            running_correct += (preds.argmax(dim=1) == y_batch).sum()
        _synchronize(device)
        seconds = time.perf_counter() - started

        test_loss, test_acc = evaluate(model, X_test, y_test, device)
        history["train_loss"].append(float(running_loss) / len(X_train))
        history["train_acc"].append(float(running_correct) / len(X_train))
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["seconds"].append(seconds)
        log(f"  epoch {epoch + 1:>2}: train loss {history['train_loss'][-1]:.4f}  "
            f"test loss {test_loss:.4f}  test acc {test_acc:.4f}  ({seconds:.1f} s)")
    return history


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def predict_proba(model, images):
    model.eval()
    return torch.softmax(model(images), dim=1)


def layer_timings(model, input_shape, device, repeats=30):
    shapes = {}

    def remember(module, inputs, output):
        shapes[module] = inputs[0].shape

    leaves = [(name, module) for name, module in model.named_modules()
              if not list(module.children()) and not isinstance(module, nn.Dropout)]
    hooks = [module.register_forward_hook(remember) for _, module in leaves]
    model = model.to(device).eval()
    with torch.no_grad():
        model(torch.zeros(input_shape, device=device))
    for hook in hooks:
        hook.remove()

    model.train()
    rows = []
    for name, module in leaves:
        x = torch.randn(shapes[module], device=device, requires_grad=True)
        for _ in range(3):
            module(x).sum().backward()
        _synchronize(device)
        started = time.perf_counter()
        for _ in range(repeats):
            module(x).sum().backward()
        _synchronize(device)
        milliseconds = (time.perf_counter() - started) / repeats * 1000
        weights = sum(parameter.numel() for parameter in module.parameters())
        rows.append({"layer": name, "kind": type(module).__name__,
                     "input": list(shapes[module][1:]), "params": weights,
                     "ms": milliseconds})
    return rows


def runs_needed(std, difference, z_alpha=1.96, z_beta=0.84):
    return int(np.ceil(2 * (z_alpha + z_beta) ** 2 * std ** 2 / difference ** 2))


def solve_padding(size, kernel, target):
    padding = []
    for n, k, out in zip(size, kernel, target):
        doubled = out - n + k - 1
        if doubled < 0 or doubled % 2:
            raise ValueError(f"no symmetric padding maps {n} to {out} with k={k}")
        padding.append(doubled // 2)
    return tuple(padding)


PADDING_TASKS = [((3, 3), (8, 16)), ((5, 5), (8, 16)), ((7, 7), (8, 16)),
                 ((9, 9), (8, 16)), ((3, 5), (8, 16)), ((3, 3), (22, 30)),
                 ((4, 4), (7, 15)), ((2, 2), (9, 17))]


def padding_exercise(N=4, C=3, C_out=10, H=8, W=16):
    x = torch.ones((N, C, H, W))
    rows = []
    for kernel, target in PADDING_TASKS:
        padding = solve_padding((H, W), kernel, target)
        out = nn.Conv2d(C, C_out, kernel_size=kernel, padding=padding)(x)
        rows.append({"kernel": kernel, "padding": padding, "target": target,
                     "shape": tuple(out.shape)})
    return rows
