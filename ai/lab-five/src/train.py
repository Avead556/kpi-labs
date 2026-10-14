import json
import time
from pathlib import Path

import numpy as np
import torch

import regression

RESULTS = Path(__file__).resolve().parents[1] / "results"
EPOCHS = 20000
LEARNING_RATES = (1e-6, 1e-5, 1e-4, 1.2e-4)
MANUAL_LR = 1e-5
REPORT_EPOCHS = (0, 1, 10, 100, 500, 1000, 2000, 5000, 10000, 20000)


def first_step():
    torch.manual_seed(18)
    inputs = torch.from_numpy(regression.INPUTS)
    targets = torch.from_numpy(regression.TARGETS)
    w = torch.randn(2, 3, requires_grad=True)
    b = torch.randn(2, requires_grad=True)
    torch.set_printoptions(precision=4, sci_mode=False)

    preds = regression.model(inputs, w, b)
    loss = regression.mse(preds, targets)
    print("w =", w, "\nb =", b)
    print("preds =", preds)
    print("loss =", loss)
    loss.backward()
    print("w.grad =", w.grad, "\nb.grad =", b.grad)
    with torch.no_grad():
        w -= w.grad * MANUAL_LR
        b -= b.grad * MANUAL_LR
        w.grad.zero_()
        b.grad.zero_()
    print("loss after one step =", regression.mse(regression.model(inputs, w, b), targets))


def main():
    torch.set_num_threads(4)
    print("== one step, as in the manual ==")
    first_step()

    coef, ls_mse = regression.least_squares()
    eigen = np.linalg.eigvalsh(regression.loss_hessian())
    limit = 2.0 / eigen.max()
    print(f"\nleast squares optimum: MSE = {ls_mse:.4f}")
    print(f"Hessian eigenvalues: {', '.join(f'{v:.4g}' for v in eigen)}"
          f"  -> stable for lr < {limit:.4g}, condition number {eigen.max() / eigen.min():.3g}")

    losses, params, summary_runs = [], [], []
    for lr in LEARNING_RATES:
        started = time.perf_counter()
        run_losses, run_params = regression.train(lr, EPOCHS)
        elapsed = time.perf_counter() - started
        best = regression.optimal_epoch(run_losses)
        losses.append(run_losses)
        params.append(run_params)

        last = int(np.flatnonzero(np.isfinite(run_losses))[-1])
        print(f"\n== lr = {lr:g}  ({elapsed:.1f} s) ==")
        for epoch in REPORT_EPOCHS:
            print(f"  epoch {epoch:>6}: MSE = {run_losses[epoch]:.6g}")
        if last < EPOCHS:
            print(f"  diverged: last finite MSE {run_losses[last]:.4g} at epoch {last}")
        elif best is None:
            print(f"  no plateau within {EPOCHS} epochs: still falling")
        else:
            print(f"  optimal epochs (within {regression.PLATEAU_MARGIN:.0%} of the final MSE):"
                  f" {best}, MSE = {run_losses[best]:.4f}")
        summary_runs.append({
            "lr": lr,
            "seconds": round(elapsed, 2),
            "finalMse": float(run_losses[-1]) if last == EPOCHS else None,
            "optimalEpoch": best,
            "optimalMse": float(run_losses[best]) if best is not None else None,
            "divergedAt": None if last == EPOCHS else last,
            "milestones": {str(e): float(run_losses[e]) for e in REPORT_EPOCHS},
        })

    manual = LEARNING_RATES.index(MANUAL_LR)
    best = summary_runs[manual]["optimalEpoch"]
    final = params[manual][-1]
    print(f"\n== lr = {MANUAL_LR:g}: model after {EPOCHS} epochs ==")
    print("w =\n", final[:6].reshape(2, 3), "\nb =", final[6:])
    print("predictions:\n", np.round(regression.predict(final), 2))
    print("targets:\n", regression.TARGETS)
    print(f"model at the optimal epoch {best}:")
    print("w =\n", params[manual][best][:6].reshape(2, 3), "\nb =", params[manual][best][6:])
    print("least squares w|b =\n", coef.T)

    RESULTS.mkdir(exist_ok=True)
    np.savez_compressed(RESULTS / "regression.npz",
                        learning_rates=np.array(LEARNING_RATES),
                        losses=np.array(losses), params=np.array(params))
    summary = {
        "epochs": EPOCHS,
        "seed": 18,
        "manualLr": MANUAL_LR,
        "stabilityLimit": limit,
        "hessianEigenvalues": eigen.tolist(),
        "leastSquares": {"mse": ls_mse, "coef": coef.T.tolist()},
        "runs": summary_runs,
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nsaved {RESULTS / 'regression.npz'} and summary.json")


if __name__ == "__main__":
    main()
