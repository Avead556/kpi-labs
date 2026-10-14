import time

import numpy as np
import torch

ACTIVATIONS = ("ELU", "Hardtanh", "LeakyReLU", "LogSigmoid", "PReLU", "ReLU", "ReLU6",
               "RReLU", "SELU", "CELU", "Sigmoid", "Softplus", "Softshrink", "Softsign",
               "Tanh", "Tanhshrink", "Hardshrink")
NUMBER_OF_EXPERIMENTS = 200


class SimpleNet(torch.nn.Module):

    def __init__(self, activation):
        super().__init__()
        self.activation = activation
        self.fc1 = torch.nn.Linear(1, 1, bias=False)
        self.fc1.weight.data.fill_(1.0)
        self.fc2 = torch.nn.Linear(1, 1, bias=False)
        self.fc2.weight.data.fill_(1.0)
        self.fc3 = torch.nn.Linear(1, 1, bias=False)
        self.fc3.weight.data.fill_(1.0)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        return x


def fc1_gradient(net, x):
    net.forward(x).sum().backward()
    value = torch.abs(net.fc1.weight.grad).item()
    net.zero_grad()
    return value


def gradient_probe(seed, experiments=NUMBER_OF_EXPERIMENTS):
    results = {}
    for name in ACTIVATIONS:
        np.random.seed(seed)
        torch.manual_seed(seed)
        net = SimpleNet(activation=getattr(torch.nn, name)())
        grads = [fc1_gradient(net, x) for x in torch.randn((experiments, 1))]
        results[name] = float(np.mean(grads))
    return results


def _activation(name):
    if name == "tanh":
        return torch.nn.Tanh()
    if name == "relu":
        return torch.nn.ReLU()
    raise NotImplementedError(name)


def _pooling(name):
    if name == "avg":
        return torch.nn.AvgPool2d(kernel_size=2, stride=2)
    if name == "max":
        return torch.nn.MaxPool2d(kernel_size=2, stride=2)
    raise NotImplementedError(name)


def _conv_block(in_channels, out_channels, conv_size, padding):
    if conv_size == 5:
        return torch.nn.Conv2d(in_channels, out_channels, 5, padding=2 if padding else 0)
    if conv_size == 3:
        pad = 1 if padding else 0
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, padding=pad),
            torch.nn.Conv2d(out_channels, out_channels, 3, padding=pad))
    raise NotImplementedError(conv_size)


class LeNet5(torch.nn.Module):

    def __init__(self, activation="tanh", pooling="avg", conv_size=5, use_batch_norm=False,
                 in_channels=1, image_size=28, channels=(6, 16), pad_all=False,
                 input_batch_norm=False):
        super().__init__()
        c1, c2 = channels
        self.use_batch_norm = use_batch_norm
        self.bn0 = torch.nn.BatchNorm2d(in_channels) if input_batch_norm else torch.nn.Identity()

        self.conv1 = _conv_block(in_channels, c1, conv_size, padding=True)
        self.act1 = _activation(activation)
        self.bn1 = torch.nn.BatchNorm2d(c1)
        self.pool1 = _pooling(pooling)

        self.conv2 = _conv_block(c1, c2, conv_size, padding=pad_all)
        self.act2 = _activation(activation)
        self.bn2 = torch.nn.BatchNorm2d(c2)
        self.pool2 = _pooling(pooling)

        side = image_size // 2
        side = side // 2 if pad_all else (side - 4) // 2
        self.fc1 = torch.nn.Linear(side * side * c2, 120)
        self.act3 = _activation(activation)
        self.fc2 = torch.nn.Linear(120, 84)
        self.act4 = _activation(activation)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.bn0(x)
        x = self.act1(self.conv1(x))
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.pool1(x)

        x = self.act2(self.conv2(x))
        if self.use_batch_norm:
            x = self.bn2(x)
        x = self.pool2(x)

        x = x.flatten(start_dim=1)
        x = self.act3(self.fc1(x))
        x = self.act4(self.fc2(x))
        return self.fc3(x)


class CifarNet(torch.nn.Module):

    def __init__(self, batch_norm=True, dropout=0.0, conv_dropout=0.0):
        super().__init__()

        def norm2d(channels):
            return torch.nn.BatchNorm2d(channels) if batch_norm else torch.nn.Identity()

        def norm1d(features):
            return torch.nn.BatchNorm1d(features) if batch_norm else torch.nn.Identity()

        def block(in_channels, out_channels, drop):
            layers = [torch.nn.Dropout2d(drop)] if drop else []
            layers += [torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
                       torch.nn.ReLU(), norm2d(out_channels), torch.nn.MaxPool2d(2)]
            return layers

        self.features = torch.nn.Sequential(
            norm2d(3),
            *block(3, 16, 0.0),
            *block(16, 32, conv_dropout),
            *block(32, 64, conv_dropout),
            torch.nn.Flatten())

        head = [torch.nn.Dropout(dropout)] if dropout else []
        head += [torch.nn.Linear(4 * 4 * 64, 256), torch.nn.Tanh(), norm1d(256)]
        head += [torch.nn.Dropout(dropout)] if dropout else []
        head += [torch.nn.Linear(256, 64), torch.nn.Tanh(), norm1d(64),
                 torch.nn.Linear(64, 10)]
        self.classifier = torch.nn.Sequential(*head)

    def forward(self, x):
        return self.classifier(self.features(x))


def count_parameters(net):
    return sum(p.numel() for p in net.parameters())


def evaluate(net, X, y, loss, batch_size=1000):
    net.eval()
    total_loss, predictions = 0.0, []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            logits = net(X[start:start + batch_size])
            total_loss += loss(logits, y[start:start + batch_size]).item() * len(logits)
            predictions.append(logits.argmax(dim=1))
    predictions = torch.cat(predictions)
    return total_loss / len(X), (predictions == y).float().mean().item(), predictions


def train(net, X_train, y_train, X_test, y_test, epochs, lr=1.0e-3, weight_decay=0.0,
          batch_size=100, probe=None, device="cpu", log=print):
    net = net.to(device)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    history = {key: [] for key in ("train_loss", "train_acc", "test_loss", "test_acc",
                                   "probe", "seconds")}

    for epoch in range(epochs):
        started = time.monotonic()
        order = torch.randperm(len(X_train), device=device)
        net.train()
        seen, loss_sum, correct = 0, 0.0, 0
        for start in range(0, len(X_train), batch_size):
            batch = order[start:start + batch_size]
            X_batch, y_batch = X_train[batch], y_train[batch]

            optimizer.zero_grad()
            preds = net(X_batch)
            loss_value = loss(preds, y_batch)
            loss_value.backward()
            optimizer.step()

            seen += len(batch)
            loss_sum += loss_value.detach() * len(batch)
            correct += (preds.argmax(dim=1) == y_batch).sum()

        test_loss, test_acc, predictions = evaluate(net, X_test, y_test, loss)
        history["train_loss"].append(loss_sum.item() / seen)
        history["train_acc"].append(correct.item() / seen)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        if probe is not None:
            with torch.no_grad():
                history["probe"].append(torch.softmax(net(probe), dim=1).cpu().numpy())
        history["seconds"].append(time.monotonic() - started)
        log(f"  epoch {epoch + 1:2d}: train loss {history['train_loss'][-1]:.4f} "
            f"acc {history['train_acc'][-1]:.4f} | test loss {test_loss:.4f} "
            f"acc {test_acc:.4f} | {history['seconds'][-1]:.1f}s")

    history["predictions"] = predictions.cpu().numpy()
    return history
