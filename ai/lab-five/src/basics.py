import numpy as np
import torch

import gradient


def main():
    torch.set_num_threads(4)
    x = torch.tensor([[1., 2., 3., 4.],
                      [5., 6., 7., 8.],
                      [9., 10., 11., 12.]], requires_grad=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = x.to(device)

    function = 10 * (x ** 2).sum()
    function.backward()
    print(x.grad, "<- gradient")

    node = function.grad_fn
    while node is not None:
        print(type(node).__name__)
        node = node.next_functions[0][0] if node.next_functions else None

    x = torch.tensor([8., 8.], requires_grad=True)
    var_history, fn_history = [], []
    for _ in range(500):
        var_history.append(x.data.cpu().numpy().copy())
        fn_history.append(gradient.function_parabola(x).item())
        gradient.make_gradient_step(gradient.function_parabola, x)
    manual_path = np.array(var_history)

    sgd_path, sgd_values, _ = gradient.descend("parabola", [8., 8.], 0.001, 499)
    print(f"\nparabola, lr = 0.001, 500 steps from (8, 8)")
    print(f"  hand-written: x = {manual_path[-1]}, f = {fn_history[-1]:.6g}")
    print(f"  torch.optim.SGD: x = {sgd_path[-1]}, f = {sgd_values[-1]:.6g}")
    print(f"  largest difference between the two paths: {np.abs(manual_path - sgd_path).max():.3g}")
    ratio = sgd_values[1:] / sgd_values[:-1]
    print(f"  f shrinks by {ratio.mean():.6f} per step (theory: (1 - 20 lr)^2 = {(1 - 0.02) ** 2:.6f})")

    for name in ("parabola", "skewed"):
        path, values, diverged = gradient.descend(name, [8., 8.], 0.001, 500)
        eigen = np.linalg.eigvalsh(gradient.hessian(name))
        print(f"\n{name}: Hessian eigenvalues {eigen.round(3)}, "
              f"SGD stable for lr < {gradient.sgd_stability_limit(name):.4g}")
        reached = gradient.steps_to_tolerance(values)
        print(f"  500 SGD steps, lr = 0.001: x = {path[-1]}, f = {values[-1]:.4g}, "
              + (f"f < {gradient.TOLERANCE:g} after {reached} steps" if reached is not None
                 else f"f < {gradient.TOLERANCE:g} not reached"))


if __name__ == "__main__":
    main()
