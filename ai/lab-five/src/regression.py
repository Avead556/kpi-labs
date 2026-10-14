import numpy as np
import torch

INPUTS = np.array([[73, 67, 43],
                   [91, 88, 64],
                   [87, 134, 58],
                   [102, 43, 37],
                   [69, 96, 70]], dtype="float32")

TARGETS = np.array([[56, 70],
                    [81, 101],
                    [119, 133],
                    [22, 37],
                    [103, 119]], dtype="float32")

FEATURES = ("temp", "rainfall", "humidity")
OUTPUTS = ("apples", "oranges")

PLATEAU_MARGIN = 0.01
PLATEAU_WINDOW = 1000


def model(x, w, b):
    return x @ w.t() + b


def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()


def train(lr, epochs, seed=18):
    torch.manual_seed(seed)
    inputs = torch.from_numpy(INPUTS)
    targets = torch.from_numpy(TARGETS)
    w = torch.randn(2, 3, requires_grad=True)
    b = torch.randn(2, requires_grad=True)

    losses = np.full(epochs + 1, np.nan)
    params = np.full((epochs + 1, 8), np.nan, dtype=np.float32)
    for epoch in range(epochs + 1):
        preds = model(inputs, w, b)
        loss = mse(preds, targets)
        losses[epoch] = loss.item()
        params[epoch] = torch.cat([w.detach().flatten(), b.detach()]).numpy()
        if epoch == epochs or not np.isfinite(losses[epoch]):
            break
        loss.backward()
        with torch.no_grad():
            w -= w.grad * lr
            b -= b.grad * lr
            w.grad.zero_()
            b.grad.zero_()
    return losses, params


def predict(params):
    w = params[:6].reshape(2, 3)
    return INPUTS @ w.T + params[6:]


def optimal_epoch(losses, margin=PLATEAU_MARGIN, window=PLATEAU_WINDOW):
    final = losses[-1]
    if not np.isfinite(final) or losses[-1 - window] > final * (1 + margin):
        return None
    return int(np.flatnonzero(losses <= final * (1 + margin))[0])


def least_squares():
    design = np.hstack([INPUTS, np.ones((len(INPUTS), 1))]).astype(np.float64)
    coef = np.linalg.lstsq(design, TARGETS.astype(np.float64), rcond=None)[0]
    return coef, float(((design @ coef - TARGETS) ** 2).mean())


def loss_hessian():
    design = np.hstack([INPUTS, np.ones((len(INPUTS), 1))]).astype(np.float64)
    return 2.0 / TARGETS.size * design.T @ design
