import math
import pickle
import random
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
import torchvision

DATA_ROOT = Path(__file__).resolve().parents[2] / ".data"
CIFAR_ARCHIVE = DATA_ROOT / "cifar-10-python.tar.gz"
VALIDATION_SIZE = 10_000
BATCH_SIZE = 100

OPTIMIZERS = {
    "SGD": (lambda p, lr: torch.optim.SGD(p, lr=lr), 0.1),
    "Momentum": (lambda p, lr: torch.optim.SGD(p, lr=lr, momentum=0.9), 0.01),
    "Nesterov": (lambda p, lr: torch.optim.SGD(p, lr=lr, momentum=0.9, nesterov=True), 0.01),
    "Adagrad": (lambda p, lr: torch.optim.Adagrad(p, lr=lr), 0.01),
    "RMSprop": (lambda p, lr: torch.optim.RMSprop(p, lr=lr), 1e-3),
    "Adam": (lambda p, lr: torch.optim.Adam(p, lr=lr), 1e-3),
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class Split:

    def __init__(self, x_train, y_train, x_test, y_test, classes):
        cut = len(x_train) - VALIDATION_SIZE
        self.x_train, self.y_train = x_train[:cut], y_train[:cut]
        self.x_val, self.y_val = x_train[cut:], y_train[cut:]
        self.x_test, self.y_test = x_test, y_test
        self.classes = classes

    @property
    def features(self):
        return self.x_train.shape[1]


def load_mnist(normalize=True):
    train = torchvision.datasets.MNIST(DATA_ROOT, train=True, download=False)
    test = torchvision.datasets.MNIST(DATA_ROOT, train=False, download=False)
    scale = 255.0 if normalize else 1.0
    x_train = train.data.float().reshape([-1, 28 * 28]) / scale
    x_test = test.data.float().reshape([-1, 28 * 28]) / scale
    return Split(x_train, train.targets, x_test, test.targets,
                 [str(digit) for digit in range(10)])


def load_cifar10():
    images, labels = {"train": [], "test": []}, {"train": [], "test": []}
    with tarfile.open(CIFAR_ARCHIVE, "r:gz") as archive:
        for member in archive.getmembers():
            name = Path(member.name).name
            if name.startswith("data_batch_"):
                part = "train"
            elif name == "test_batch":
                part = "test"
            elif name == "batches.meta":
                meta = pickle.load(archive.extractfile(member), encoding="latin1")
                continue
            else:
                continue
            batch = pickle.load(archive.extractfile(member), encoding="latin1")
            images[part].append((member.name, batch["data"]))
            labels[part].append((member.name, batch["labels"]))

    def stack(part):
        pixels = np.concatenate([data for _, data in sorted(images[part])])
        targets = np.concatenate([data for _, data in sorted(labels[part])])
        channel_last = pixels.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        x = torch.from_numpy(np.ascontiguousarray(channel_last)).float() / 255.0
        x = x.permute(0, 3, 1, 2)
        return x.reshape(len(x), -1), torch.from_numpy(targets).long()

    x_train, y_train = stack("train")
    x_test, y_test = stack("test")
    return Split(x_train, y_train, x_test, y_test, meta["label_names"])


class ScratchLinear(torch.nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.empty(out_features))
        bound = 1.0 / math.sqrt(in_features)
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)
            self.bias.uniform_(-bound, bound)

    def forward(self, x):
        return x @ self.weight.T + self.bias


class FCNet(torch.nn.Module):

    ACTIVATIONS = {"sigmoid": torch.nn.Sigmoid, "relu": torch.nn.ReLU}

    def __init__(self, in_features, hidden=(100,), activation="sigmoid",
                 classes=10, linear=torch.nn.Linear):
        super().__init__()
        layers, width = [], in_features
        for size in hidden:
            layers += [linear(width, size), self.ACTIVATIONS[activation]()]
            width = size
        layers.append(linear(width, classes))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


@torch.no_grad()
def evaluate(net, x, y, loss_fn, chunk=10_000):
    net.eval()
    total_loss, correct = 0.0, 0
    for start in range(0, len(x), chunk):
        scores = net(x[start:start + chunk])
        total_loss += loss_fn(scores, y[start:start + chunk]).item() * len(scores)
        correct += (scores.argmax(dim=1) == y[start:start + chunk]).sum().item()
    net.train()
    return total_loss / len(x), correct / len(x)


def train(net, data, optimizer, epochs, keep_best=False, log=None):
    loss_fn = torch.nn.CrossEntropyLoss()
    history = {key: [] for key in ("train_loss", "train_acc", "val_loss", "val_acc",
                                   "test_loss", "test_acc", "seconds")}
    best = {"epoch": 0, "val_loss": math.inf, "state": None}

    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        order = np.random.permutation(len(data.x_train))
        order = torch.from_numpy(order).to(data.x_train.device)
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            loss_value = loss_fn(net(data.x_train[batch]), data.y_train[batch])
            loss_value.backward()
            optimizer.step()
        history["seconds"].append(time.perf_counter() - started)

        for part in ("train", "val", "test"):
            loss, accuracy = evaluate(net, getattr(data, f"x_{part}"),
                                      getattr(data, f"y_{part}"), loss_fn)
            history[f"{part}_loss"].append(loss)
            history[f"{part}_acc"].append(accuracy)

        if keep_best and history["val_loss"][-1] < best["val_loss"]:
            best = {"epoch": epoch, "val_loss": history["val_loss"][-1],
                    "state": {k: v.detach().clone() for k, v in net.state_dict().items()}}
        if log:
            log(f"  epoch {epoch:3d}  train {history['train_loss'][-1]:.4f}"
                f"  val {history['val_loss'][-1]:.4f}  val acc {history['val_acc'][-1]:.4f}"
                f"  test acc {history['test_acc'][-1]:.4f}")
    return history, best


def make_optimizer(name, parameters):
    factory, lr = OPTIMIZERS[name]
    return factory(parameters, lr)


def linear_by_matmul():
    x = torch.tensor([[10., 20.]])
    fc = torch.nn.Linear(2, 3)
    w = torch.tensor([[11., 12.], [21., 22.], [31., 32.]])
    b = torch.tensor([[31., 32., 33.]])
    fc.weight.data = w
    fc.bias.data = b

    fc_out = fc(x)
    fc_out.sum().backward()
    weight_grad, bias_grad = fc.weight.grad, fc.bias.grad

    w = w.clone().requires_grad_(True)
    b = b.clone().requires_grad_(True)
    fc_out_alternative = x @ w.T + b
    our_formula = fc_out_alternative.sum()
    our_formula.backward()

    return {
        "fc_out": fc_out,
        "fc_out_alternative": fc_out_alternative,
        "equal": fc_out == fc_out_alternative,
        "fc_weight_grad": weight_grad,
        "our_weight_grad": w.grad,
        "fc_bias_grad": bias_grad,
        "our_bias_grad": b.grad,
    }
