import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import datasets
from regression import (RegressionNet, Standardizer, fit, predict, regression_metrics,
                        sine_dataset, split_indices)

SEED = 18
RESULTS = Path(__file__).resolve().parents[1] / "results"

SNAPSHOT_EVERY = 10
TABULAR_SNAPSHOT_EVERY = 20
GRID = torch.linspace(-10, 10, 400).unsqueeze(1)

BASE = {"hidden": (50,), "activation": "tanh", "loss": "mse", "epochs": 2000, "lr": 0.01}

EXPERIMENTS = {
    "neurons": ("a) кількість нейронів",
                [(f"{n}", {"hidden": (n,)}) for n in range(1, 11)]),
    "epochs": ("b) кількість епох",
               [(f"{n}", {"epochs": n}) for n in (50, 200, 500, 1000, 2000, 5000)]),
    "activation": ("c) функція активації",
                   [(name, {"activation": name})
                    for name in ("tanh", "relu", "sigmoid", "elu", "gelu")]),
    "layers": ("d) кількість прихованих шарів",
               [("0", {"hidden": ()}), ("1×50", {"hidden": (50,)}),
                ("2×50", {"hidden": (50, 50)}), ("3×50", {"hidden": (50, 50, 50)})]),
    "loss": ("e) функція втрат",
             [(name, {"loss": name}) for name in ("mse", "mae", "huber")]),
}

TABULAR_EPOCHS = 2000
ARCHITECTURES = [(), (16,), (64,), (64, 32), (128, 64, 32)]


class Tee:

    def __init__(self):
        self.lines = []

    def __call__(self, *parts):
        line = " ".join(str(part) for part in parts)
        print(line, flush=True)
        self.lines.append(line)


log = Tee()


def seeded_net(**kwargs):
    torch.manual_seed(SEED)
    return RegressionNet(**kwargs)


def train_sine(data):
    x_train, y_train, x_val, y_val = data
    net = seeded_net(n_inputs=1, hidden=BASE["hidden"], activation=BASE["activation"])
    log(net)
    started = time.perf_counter()
    probe = torch.cat([GRID, x_train, x_val])
    history = fit(net, x_train, y_train, x_val, y_val, epochs=BASE["epochs"], lr=BASE["lr"],
                  snapshot_every=SNAPSHOT_EVERY, snapshot_x=probe)
    seconds = time.perf_counter() - started
    snapshots = np.stack(history["snapshots"])[:, :, 0].astype(np.float32)
    n_grid, n_train = len(GRID), len(x_train)

    for epoch in (1, 10, 100, 500, 1000, 2000):
        log(f"  epoch {epoch:>4}: train MSE {history['train_loss'][epoch - 1]:.4f}, "
            f"validation MSE {history['validation_mse'][epoch - 1]:.4f}, "
            f"MAE {history['validation_mae'][epoch - 1]:.4f}")
    log(f"  trained in {seconds:.2f} s")

    np.savez_compressed(
        RESULTS / "sine.npz",
        x_train=x_train.numpy().ravel(), y_train=y_train.numpy().ravel(),
        x_validation=x_val.numpy().ravel(), y_validation=y_val.numpy().ravel(),
        grid=GRID.numpy().ravel(),
        snapshot_epochs=np.array(history["snapshot_epochs"]),
        snapshots=snapshots[:, :n_grid],
        train_snapshots=snapshots[:, n_grid:n_grid + n_train],
        validation_snapshots=snapshots[:, n_grid + n_train:],
        train_loss=np.array(history["train_loss"], dtype=np.float32),
        validation_mse=np.array(history["validation_mse"], dtype=np.float32),
        validation_mae=np.array(history["validation_mae"], dtype=np.float32),
    )
    return {"seconds": round(seconds, 2),
            "final": {"train_mse": history["train_loss"][-1],
                      "validation_mse": history["validation_mse"][-1],
                      "validation_mae": history["validation_mae"][-1]},
            "noise_mse": float(((y_train - torch.sin(x_train)) ** 2).mean())}


def run_variant(data, config):
    x_train, y_train, x_val, y_val = data
    net = seeded_net(n_inputs=1, hidden=config["hidden"], activation=config["activation"])
    started = time.perf_counter()
    history = fit(net, x_train, y_train, x_val, y_val, epochs=config["epochs"],
                  lr=config["lr"], loss=config["loss"])
    seconds = time.perf_counter() - started
    return net, history, seconds


def train_experiments(data):
    arrays, table = {}, {}
    for key, (title, variants) in EXPERIMENTS.items():
        log(f"\n{title}")
        rows, predictions, curves = [], [], []
        if key == "epochs":
            longest = max(v["epochs"] for _, v in variants)
            net = seeded_net(n_inputs=1, hidden=BASE["hidden"], activation=BASE["activation"])
            history = fit(net, *data, epochs=longest, lr=BASE["lr"],
                          snapshot_every=50, snapshot_x=GRID)
            by_epoch = dict(zip(history["snapshot_epochs"], history["snapshots"]))
            for label, change in variants:
                n = change["epochs"]
                rows.append({"label": label, "validation_mse": history["validation_mse"][n - 1],
                             "validation_mae": history["validation_mae"][n - 1],
                             "train_loss": history["train_loss"][n - 1]})
                predictions.append(by_epoch[n][:, 0])
            curves = [history["train_loss"]]
        else:
            for label, change in variants:
                net, history, seconds = run_variant(data, {**BASE, **change})
                rows.append({"label": label, "validation_mse": history["validation_mse"][-1],
                             "validation_mae": history["validation_mae"][-1],
                             "train_loss": history["train_loss"][-1],
                             "parameters": sum(p.numel() for p in net.parameters()),
                             "seconds": round(seconds, 2)})
                predictions.append(predict(net, GRID)[:, 0])
                curves.append(history["train_loss"])
        for row in rows:
            log(f"  {row['label']:>8}: validation MSE {row['validation_mse']:.4f}, "
                f"MAE {row['validation_mae']:.4f}, final train loss {row['train_loss']:.4f}"
                + (f", {row['parameters']} parameters" if "parameters" in row else ""))
        table[key] = {"title": title, "rows": rows}
        arrays[f"{key}_predictions"] = np.stack(predictions).astype(np.float32)
        arrays[f"{key}_curves"] = np.array(curves, dtype=np.float32)
    np.savez_compressed(RESULTS / "experiments.npz", **arrays)
    return table


def to_tensor(values):
    return torch.tensor(np.asarray(values), dtype=torch.float32)


def train_tabular(name, frame, targets, feature_sets, lr=0.01, weight_decay=0.0):
    train_rows, val_rows, test_rows = split_indices(len(frame), seed=SEED)
    log(f"\n{name}: {len(frame)} rows -> train {len(train_rows)}, "
        f"validation {len(val_rows)}, test {len(test_rows)}; Adam lr {lr}, L2 {weight_decay}")

    y = frame[targets].to_numpy()
    y_scale = Standardizer(y[train_rows])
    y_train, y_val = to_tensor(y_scale(y[train_rows])), to_tensor(y_scale(y[val_rows]))

    search = []
    for set_name, features in feature_sets.items():
        x = frame[features].to_numpy()
        x_scale = Standardizer(x[train_rows])
        x_train, x_val, x_test = (to_tensor(x_scale(x[rows]))
                                  for rows in (train_rows, val_rows, test_rows))
        for hidden in ARCHITECTURES:
            net = seeded_net(n_inputs=len(features), hidden=hidden, activation="relu",
                             n_outputs=len(targets))
            history = fit(net, x_train, y_train, x_val, y_val,
                          epochs=TABULAR_EPOCHS, lr=lr, weight_decay=weight_decay)
            best = history["best_epoch"]
            net.load_state_dict(history["best_state"])
            test = regression_metrics(y_scale.inverse(predict(net, x_test)), y[test_rows])
            search.append({"features": set_name, "hidden": list(hidden),
                           "best_epoch": best,
                           "validation_mse_z": history["validation_mse"][best - 1],
                           "final_validation_mse_z": history["validation_mse"][-1],
                           "test_mae": test["mae"].tolist(), "test_r2": test["r2"].tolist()})
            log(f"  {set_name:<14} hidden {str(list(hidden)):<14} "
                f"validation MSE(z) best {search[-1]['validation_mse_z']:.4f} at epoch {best:>4}, "
                f"last {search[-1]['final_validation_mse_z']:.4f} | test R2 "
                + " ".join(f"{r:.3f}" for r in test["r2"]))

    chosen = min(search, key=lambda row: row["validation_mse_z"])
    features = feature_sets[chosen["features"]]
    log(f"  chosen: {chosen['features']}, hidden {chosen['hidden']}")

    x = frame[features].to_numpy()
    x_scale = Standardizer(x[train_rows])
    x_train, x_val, x_test = (to_tensor(x_scale(x[rows])) for rows in (train_rows, val_rows, test_rows))
    net = seeded_net(n_inputs=len(features), hidden=tuple(chosen["hidden"]), activation="relu",
                     n_outputs=len(targets))
    started = time.perf_counter()
    history = fit(net, x_train, y_train, x_val, y_val, epochs=TABULAR_EPOCHS, lr=lr,
                  weight_decay=weight_decay, snapshot_every=TABULAR_SNAPSHOT_EVERY, snapshot_x=x_test)
    seconds = time.perf_counter() - started
    net.load_state_dict(history["best_state"])

    test_prediction = y_scale.inverse(predict(net, x_test))
    test_metrics = regression_metrics(test_prediction, y[test_rows])
    val_metrics = regression_metrics(y_scale.inverse(predict(net, x_val)), y[val_rows])
    baseline = regression_metrics(np.broadcast_to(y[train_rows].mean(axis=0), y[test_rows].shape),
                                  y[test_rows])
    for i, target in enumerate(targets):
        log(f"  {target}: test MSE {test_metrics['mse'][i]:.3f}, MAE {test_metrics['mae'][i]:.3f}, "
            f"R2 {test_metrics['r2'][i]:.4f} (validation MAE {val_metrics['mae'][i]:.3f}; "
            f"predicting the train mean: MAE {baseline['mae'][i]:.3f})")
    log(f"  best epoch {history['best_epoch']}, retrained in {seconds:.2f} s")

    snapshots = np.stack([y_scale.inverse(s) for s in history["snapshots"]]).astype(np.float32)
    arrays = {
        f"{name}_snapshot_epochs": np.array(history["snapshot_epochs"]),
        f"{name}_snapshots": snapshots,
        f"{name}_test_target": y[test_rows].astype(np.float32),
        f"{name}_train_loss": np.array(history["train_loss"], dtype=np.float32),
        f"{name}_validation_mse": np.array(history["validation_mse"], dtype=np.float32),
        f"{name}_best_epoch": np.array(history["best_epoch"]),
    }
    summary = {
        "rows": {"train": len(train_rows), "validation": len(val_rows), "test": len(test_rows)},
        "targets": targets, "feature_sets": feature_sets, "search": search,
        "chosen": {**chosen, "features_list": features},
        "best_epoch": history["best_epoch"], "seconds": round(seconds, 2),
        "lr": lr, "weight_decay": weight_decay, "epochs": TABULAR_EPOCHS,
        "test": {k: v.tolist() for k, v in test_metrics.items()},
        "validation": {k: v.tolist() for k, v in val_metrics.items()},
        "mean_baseline_test": {k: v.tolist() for k, v in baseline.items()},
        "correlation_with_target": frame.corr()[targets].round(3).to_dict(),
    }
    return arrays, summary


def main():
    torch.set_num_threads(4)
    RESULTS.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    summary = {"seed": SEED, "torch": torch.__version__}

    log("part 1: SineNet on noisy sin(x)")
    data = sine_dataset(seed=SEED)
    summary["sine"] = train_sine(data)
    summary["experiments"] = train_experiments(data)

    energy = datasets.load_energy()
    diabetes = datasets.load_diabetes()
    log("\ncorrelation with the targets:")
    log(energy.corr()[datasets.ENERGY_TARGETS].round(2).to_string())
    log(diabetes.corr()["progression"].round(2).to_string())

    all_energy = datasets.ENERGY_FEATURES
    energy_sets = {
        "усі 8": all_energy,
        "без X6, X8": [f for f in all_energy if f not in ("X6", "X8")],
        "без X1, X6, X8": [f for f in all_energy if f not in ("X1", "X6", "X8")],
    }
    all_diabetes = [c for c in diabetes.columns if c != "progression"]
    diabetes_sets = {
        "усі 10": all_diabetes,
        "без sex": [c for c in all_diabetes if c != "sex"],
        "|r| > 0.3": [c for c in all_diabetes if abs(diabetes.corr()["progression"][c]) > 0.3],
    }

    tabular = {}
    arrays, summary["energy"] = train_tabular("energy", energy, datasets.ENERGY_TARGETS, energy_sets)
    tabular.update(arrays)
    arrays, summary["diabetes"] = train_tabular("diabetes", diabetes, ["progression"],
                                                diabetes_sets, lr=0.001, weight_decay=0.1)
    tabular.update(arrays)
    np.savez_compressed(RESULTS / "tabular.npz", **tabular)

    summary["total_seconds"] = round(time.perf_counter() - started, 1)
    log(f"\nall done in {summary['total_seconds']} s")
    (RESULTS / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (RESULTS / "train_log.txt").write_text("\n".join(log.lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
