import copy

import numpy as np
import torch

ACTIVATIONS = {
    "tanh": torch.nn.Tanh,
    "relu": torch.nn.ReLU,
    "sigmoid": torch.nn.Sigmoid,
    "elu": torch.nn.ELU,
    "gelu": torch.nn.GELU,
}


def mse(prediction, target):
    squares = (prediction - target) ** 2
    return squares.mean()


def mae(prediction, target):
    return (prediction - target).abs().mean()


def huber(prediction, target):
    return torch.nn.functional.smooth_l1_loss(prediction, target)


LOSSES = {"mse": mse, "mae": mae, "huber": huber}


class RegressionNet(torch.nn.Module):

    def __init__(self, n_inputs=1, hidden=(50,), activation="tanh", n_outputs=1):
        super().__init__()
        layers = []
        width = n_inputs
        for n_neurons in hidden:
            layers += [torch.nn.Linear(width, n_neurons), ACTIVATIONS[activation]()]
            width = n_neurons
        layers.append(torch.nn.Linear(width, n_outputs))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def sine_dataset(n_points=100, noise=0.2, seed=18):
    generator = torch.Generator().manual_seed(seed)
    x_train = torch.rand(n_points, generator=generator) * 20.0 - 10.0
    y_train = torch.sin(x_train) + torch.randn(n_points, generator=generator) * noise
    x_validation = torch.linspace(-10, 10, n_points)
    y_validation = torch.sin(x_validation)
    return (x_train.unsqueeze(1), y_train.unsqueeze(1),
            x_validation.unsqueeze(1), y_validation.unsqueeze(1))


def split_indices(n_rows, shares=(0.6, 0.2, 0.2), seed=18):
    order = np.random.default_rng(seed).permutation(n_rows)
    n_train = round(n_rows * shares[0])
    n_validation = round(n_rows * shares[1])
    return order[:n_train], order[n_train:n_train + n_validation], order[n_train + n_validation:]


class Standardizer:

    def __init__(self, values):
        self.mean = values.mean(axis=0)
        self.std = values.std(axis=0)
        self.std[self.std == 0] = 1.0

    def __call__(self, values):
        return (values - self.mean) / self.std

    def inverse(self, values):
        return values * self.std + self.mean


def regression_metrics(prediction, target):
    prediction = np.asarray(prediction, dtype=np.float64).reshape(len(target), -1)
    target = np.asarray(target, dtype=np.float64).reshape(len(target), -1)
    error = prediction - target
    residual = (error ** 2).sum(axis=0)
    total = ((target - target.mean(axis=0)) ** 2).sum(axis=0)
    return {
        "mse": (error ** 2).mean(axis=0),
        "mae": np.abs(error).mean(axis=0),
        "r2": 1.0 - residual / total,
    }


def fit(net, x_train, y_train, x_validation, y_validation, epochs=2000, lr=0.01,
        loss="mse", weight_decay=0.0, snapshot_every=0, snapshot_x=None):
    loss_fn = LOSSES[loss]
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"train_loss": [], "validation_mse": [], "validation_mae": [],
               "snapshot_epochs": [], "snapshots": []}
    best = (float("inf"), 0, None)

    def snapshot(epoch):
        with torch.no_grad():
            history["snapshot_epochs"].append(epoch)
            history["snapshots"].append(net(snapshot_x).numpy().copy())

    if snapshot_every:
        snapshot(0)

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        y_pred = net.forward(x_train)
        loss_value = loss_fn(y_pred, y_train)
        loss_value.backward()
        optimizer.step()

        with torch.no_grad():
            y_val = net(x_validation)
            validation_mse = mse(y_val, y_validation).item()
            history["train_loss"].append(loss_value.item())
            history["validation_mse"].append(validation_mse)
            history["validation_mae"].append(mae(y_val, y_validation).item())
        if validation_mse < best[0]:
            best = (validation_mse, epoch, copy.deepcopy(net.state_dict()))
        if snapshot_every and epoch % snapshot_every == 0:
            snapshot(epoch)

    history["best_epoch"] = best[1]
    history["best_state"] = best[2]
    return history


def predict(net, x):
    with torch.no_grad():
        return net.forward(x).numpy()
