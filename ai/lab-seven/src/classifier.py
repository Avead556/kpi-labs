import random
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import sklearn.datasets
import torch
from sklearn.model_selection import train_test_split

ACTIVATIONS = {
    "sigmoid": torch.nn.Sigmoid,
    "tanh": torch.nn.Tanh,
    "relu": torch.nn.ReLU,
    "elu": torch.nn.ELU,
    "leaky_relu": torch.nn.LeakyReLU,
}

OPTIMIZERS = {
    "adam": lambda params, lr, wd: torch.optim.Adam(params, lr=lr, weight_decay=wd),
    "adamw": lambda params, lr, wd: torch.optim.AdamW(params, lr=lr, weight_decay=wd),
    "sgd": lambda params, lr, wd: torch.optim.SGD(params, lr=lr, weight_decay=wd),
    "sgd_momentum": lambda params, lr, wd: torch.optim.SGD(params, lr=lr, momentum=0.9,
                                                           weight_decay=wd),
    "rmsprop": lambda params, lr, wd: torch.optim.RMSprop(params, lr=lr, weight_decay=wd),
}


@dataclass(frozen=True)
class Config:
    features: int | None = 2
    standardize: bool = False
    hidden: tuple = (5, 5)
    activation: str = "sigmoid"
    optimizer: str = "adam"
    lr: float = 1e-3
    batch_size: int = 10
    epochs: int = 5000
    weight_decay: float = 0.0
    dropout: float = 0.0

    def describe(self):
        width = "×".join(str(n) for n in self.hidden)
        features = "усі" if self.features is None else str(self.features)
        scaled = ", z-норм." if self.standardize else ""
        extra = ""
        if self.weight_decay:
            extra += f", wd={self.weight_decay:g}"
        if self.dropout:
            extra += f", dropout={self.dropout:g}"
        return (f"ознак: {features}{scaled}; шари {width}, {self.activation}; "
                f"{self.optimizer} lr={self.lr:g}, batch={self.batch_size}{extra}")


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


@dataclass
class Split:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list
    class_names: list
    mean: np.ndarray = field(default=None)
    std: np.ndarray = field(default=None)


def load_split(name, test_size=0.3, split_seed=0):
    loader = {"wine": sklearn.datasets.load_wine,
              "breast_cancer": sklearn.datasets.load_breast_cancer}[name]
    data = loader()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=test_size, shuffle=True,
        stratify=data.target, random_state=split_seed)
    return Split(X_train, X_test, y_train, y_test,
                 list(data.feature_names), list(data.target_names))


def select(split, columns=None, standardize=False):
    columns = list(range(split.X_train.shape[1])) if columns is None else list(columns)
    X_train, X_test = split.X_train[:, columns], split.X_test[:, columns]
    mean = np.zeros(len(columns))
    std = np.ones(len(columns))
    if standardize:
        mean, std = X_train.mean(axis=0), X_train.std(axis=0)
        X_train, X_test = (X_train - mean) / std, (X_test - mean) / std
    return Split(X_train, X_test, split.y_train, split.y_test,
                 [split.feature_names[i] for i in columns], split.class_names, mean, std)


def drop_correlated(X, names, threshold=0.9):
    corr = np.corrcoef(X, rowvar=False)
    kept, dropped = [], []
    for i in range(X.shape[1]):
        partner = next((k for k in kept if abs(corr[i, k]) > threshold), None)
        if partner is None:
            kept.append(i)
        else:
            dropped.append((names[i], names[partner], float(corr[i, partner])))
    return kept, dropped, corr


class ClassifierNet(torch.nn.Module):

    def __init__(self, n_inputs, n_classes, hidden=(5, 5), activation="sigmoid", dropout=0.0):
        super().__init__()
        layers, width = [], n_inputs
        for n in hidden:
            layers += [torch.nn.Linear(width, n), ACTIVATIONS[activation]()]
            if dropout:
                layers.append(torch.nn.Dropout(dropout))
            width = n
        layers.append(torch.nn.Linear(width, n_classes))
        self.layers = torch.nn.Sequential(*layers)
        self.sm = torch.nn.Softmax(dim=1)

    def forward(self, x):
        return self.layers(x)

    def inference(self, x):
        return self.sm(self.forward(x))

    def linear_weights(self):
        return [(m.weight.detach().numpy().copy(), m.bias.detach().numpy().copy())
                for m in self.layers if isinstance(m, torch.nn.Linear)]


def train(config, split, seed=0, log_every=10, snapshot_every=50):
    fix_seed(seed)
    X_train = torch.FloatTensor(split.X_train)
    X_test = torch.FloatTensor(split.X_test)
    y_train = torch.LongTensor(split.y_train)
    y_test = torch.LongTensor(split.y_test)

    net = ClassifierNet(X_train.shape[1], len(split.class_names), config.hidden,
                        config.activation, config.dropout)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = OPTIMIZERS[config.optimizer](net.parameters(), config.lr, config.weight_decay)

    history = {key: [] for key in ("epoch", "train_loss", "train_acc", "test_loss", "test_acc")}
    snapshots = {"epoch": [], "test_pred": [], "weights": []}

    def evaluate(epoch):
        net.eval()
        with torch.no_grad():
            train_out, test_out = net(X_train), net(X_test)
        net.train()
        if epoch % log_every == 0:
            history["epoch"].append(epoch)
            history["train_loss"].append(loss(train_out, y_train).item())
            history["train_acc"].append((train_out.argmax(1) == y_train).float().mean().item())
            history["test_loss"].append(loss(test_out, y_test).item())
            history["test_acc"].append((test_out.argmax(1) == y_test).float().mean().item())
        if epoch % snapshot_every == 0:
            snapshots["epoch"].append(epoch)
            snapshots["test_pred"].append(test_out.argmax(1).numpy().astype(np.int8))
            snapshots["weights"].append(net.linear_weights())

    started = time.perf_counter()
    evaluate(0)
    for epoch in range(1, config.epochs + 1):
        order = np.random.permutation(len(X_train))
        for start_index in range(0, len(X_train), config.batch_size):
            optimizer.zero_grad()
            batch_indexes = order[start_index:start_index + config.batch_size]
            preds = net.forward(X_train[batch_indexes])
            loss_value = loss(preds, y_train[batch_indexes])
            loss_value.backward()
            optimizer.step()
        if epoch % log_every == 0 or epoch % snapshot_every == 0:
            evaluate(epoch)

    return {
        "config": asdict(config),
        "seed": seed,
        "seconds": time.perf_counter() - started,
        "history": {key: np.asarray(value) for key, value in history.items()},
        "snapshots": snapshots,
        "final_test_acc": history["test_acc"][-1],
        "final_train_acc": history["train_acc"][-1],
    }


def confusion_matrix(y_true, y_pred, n_classes):
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(matrix, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return matrix
