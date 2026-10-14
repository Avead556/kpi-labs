import math

import numpy as np
import torch

GRAMMA = [[1.0, -1.0], [1.0, 1.0]]
STRETCH = [[1.0, 0.0], [0.0, 4.0]]

DIVERGENCE_RADIUS = 1e6
TOLERANCE = 1e-3


def function_parabola(variable):
    return 10 * (variable ** 2).sum()


def function_skewed(variable):
    gramma = (torch.tensor(GRAMMA, dtype=variable.dtype)
              @ torch.tensor(STRETCH, dtype=variable.dtype))
    return 10 * (variable.unsqueeze(0) @ (gramma @ variable.unsqueeze(1))).sum()


def parabola_np(x1, x2):
    return 10 * (x1 ** 2 + x2 ** 2)


def skewed_np(x1, x2):
    g = np.array(GRAMMA) @ np.array(STRETCH)
    return 10 * (g[0, 0] * x1 * x1 + (g[0, 1] + g[1, 0]) * x1 * x2 + g[1, 1] * x2 * x2)


def hessian(name):
    if name == "parabola":
        return 20.0 * np.eye(2)
    g = np.array(GRAMMA) @ np.array(STRETCH)
    return 10.0 * (g + g.T)


def sgd_stability_limit(name):
    return 2.0 / float(np.linalg.eigvalsh(hessian(name)).max())


FUNCTIONS = {
    "parabola": (function_parabola, parabola_np, "10·(x₁² + x₂²)"),
    "skewed": (function_skewed, skewed_np, "10·xᵀ·G·x, G = [[1, −4], [1, 4]]"),
}

OPTIMIZERS = {
    "sgd": ("SGD", lambda params, lr: torch.optim.SGD(params, lr=lr)),
    "momentum": ("SGD + momentum 0.9",
                 lambda params, lr: torch.optim.SGD(params, lr=lr, momentum=0.9)),
    "adam": ("Adam", lambda params, lr: torch.optim.Adam(params, lr=lr)),
    "rmsprop": ("RMSprop", lambda params, lr: torch.optim.RMSprop(params, lr=lr)),
}


def make_gradient_step(function, variable, lr=0.001):
    function_result = function(variable)
    function_result.backward()
    variable.data -= lr * variable.grad
    variable.grad.zero_()


def descend(function_name, start, lr, steps, optimizer="sgd"):
    function = FUNCTIONS[function_name][0]
    x = torch.tensor(start, dtype=torch.float32, requires_grad=True)
    step = OPTIMIZERS[optimizer][1]([x], lr)

    var_history, fn_history = [], []
    diverged = False
    for _ in range(steps + 1):
        value = function(x)
        var_history.append(x.data.numpy().copy())
        fn_history.append(value.item())
        if len(fn_history) == steps + 1:
            break
        if not math.isfinite(fn_history[-1]) or x.data.abs().max() > DIVERGENCE_RADIUS:
            diverged = True
            break
        value.backward()
        step.step()
        step.zero_grad()
    return np.array(var_history), np.array(fn_history), diverged


def steps_to_tolerance(fn_history, tolerance=TOLERANCE):
    above = np.flatnonzero(~(np.asarray(fn_history) < tolerance))
    if above.size == 0:
        return 0
    return int(above[-1] + 1) if above[-1] + 1 < len(fn_history) else None
