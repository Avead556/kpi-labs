import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

import classifier
from classifier import Config

RESULTS = Path(__file__).resolve().parents[1] / "results"
SEEDS = (0, 1, 2)
WORKERS = 4

BASE = Config()

WINE_EXPERIMENTS = [
    ("base", "—", "базова (посібник)", BASE),
    ("a_w10", "a", "2×10 нейронів", replace(BASE, hidden=(10, 10))),
    ("a_w20", "a", "2×20 нейронів", replace(BASE, hidden=(20, 20))),
    ("a_w50", "a", "2×50 нейронів", replace(BASE, hidden=(50, 50))),
    ("b_d1", "b", "1 прихований шар", replace(BASE, hidden=(5,))),
    ("b_d3", "b", "3 приховані шари", replace(BASE, hidden=(5, 5, 5))),
    ("b_d4", "b", "4 приховані шари", replace(BASE, hidden=(5, 5, 5, 5))),
    ("c_sgd", "c", "SGD", replace(BASE, optimizer="sgd")),
    ("c_mom", "c", "SGD + momentum 0.9", replace(BASE, optimizer="sgd_momentum")),
    ("c_rms", "c", "RMSprop", replace(BASE, optimizer="rmsprop")),
    ("c_adamw", "c", "AdamW", replace(BASE, optimizer="adamw", weight_decay=1e-2)),
    ("d_lr4", "d", "lr = 1e-4", replace(BASE, lr=1e-4)),
    ("d_lr2", "d", "lr = 1e-2", replace(BASE, lr=1e-2)),
    ("d_lr1", "d", "lr = 1e-1", replace(BASE, lr=1e-1)),
    ("e_tanh", "e", "Tanh", replace(BASE, activation="tanh")),
    ("e_relu", "e", "ReLU", replace(BASE, activation="relu")),
    ("e_elu", "e", "ELU", replace(BASE, activation="elu")),
    ("e_leaky", "e", "LeakyReLU", replace(BASE, activation="leaky_relu")),
    ("f_all", "f", "усі 13 ознак", replace(BASE, features=None)),
    ("g_std", "g", "z-нормалізація", replace(BASE, standardize=True)),
    ("g_all_std", "g", "13 ознак + z-норм.", replace(BASE, features=None, standardize=True)),
    ("g_batch32", "g", "batch = 32", replace(BASE, batch_size=32)),
    ("g_dropout", "g", "dropout 0.2, 2×20", replace(BASE, hidden=(20, 20), dropout=0.2)),
    ("h_2d_tanh", "h", "2 ознаки: z-норм., 2×20 tanh, lr 1e-2",
     replace(BASE, standardize=True, hidden=(20, 20), activation="tanh", lr=1e-2)),
    ("h_2d_relu", "h", "2 ознаки: z-норм., 2×20 ReLU",
     replace(BASE, standardize=True, hidden=(20, 20), activation="relu")),
    ("h_2d_elu_wd", "h", "2 ознаки: z-норм., 2×10 ELU, AdamW",
     replace(BASE, standardize=True, hidden=(10, 10), activation="elu",
             optimizer="adamw", weight_decay=1e-2)),
    ("h_all_relu", "h", "13 ознак: z-норм., 2×20 ReLU",
     replace(BASE, features=None, standardize=True, hidden=(20, 20), activation="relu")),
    ("h_all_1x10", "h", "13 ознак: z-норм., 1×10 ReLU, AdamW",
     replace(BASE, features=None, standardize=True, hidden=(10,), activation="relu",
             optimizer="adamw", weight_decay=1e-2)),
]

CANCER_OVERFIT = Config(features=None, standardize=True, hidden=(10,), activation="relu",
                        optimizer="adamw", lr=1e-3, weight_decay=1e-2, epochs=300)
CANCER_CONFIG = replace(CANCER_OVERFIT, lr=1e-4, weight_decay=1e-1)
CANCER_VARIANTS = {
    "all": ("30 ознак", False, CANCER_CONFIG),
    "filtered": ("ознаки після фільтра |r| > 0.9", True, CANCER_CONFIG),
    "overfit": ("30 ознак, lr 1e-3 (перенавчання)", False, CANCER_OVERFIT),
}
CANCER_LOG_EVERY = 2
CANCER_SNAPSHOT_EVERY = 5


def _init_worker():
    torch.set_num_threads(1)


def _wine_job(args):
    key, config, seed = args
    split = classifier.load_split("wine")
    columns = None if config.features is None else range(config.features)
    data = classifier.select(split, columns, config.standardize)
    run = classifier.train(config, data, seed=seed)
    run["key"] = key
    run["mean"], run["std"] = data.mean, data.std
    return run


def _cancer_job(args):
    variant, columns, seed = args
    config = CANCER_VARIANTS[variant][2]
    data = classifier.select(classifier.load_split("breast_cancer"), columns, standardize=True)
    run = classifier.train(config, data, seed=seed, log_every=CANCER_LOG_EVERY,
                           snapshot_every=CANCER_SNAPSHOT_EVERY)
    run["key"] = variant
    return run


def run_all(job, tasks):
    with ProcessPoolExecutor(WORKERS, initializer=_init_worker) as pool:
        return list(pool.map(job, tasks))


def train_wine():
    split = classifier.load_split("wine")
    tasks = [(key, config, seed) for key, _, _, config in WINE_EXPERIMENTS for seed in SEEDS]
    runs = run_all(_wine_job, tasks)
    by_key = {}
    for run in runs:
        by_key.setdefault(run["key"], {})[run["seed"]] = run

    table, curves, weights = [], {}, {}
    print(f"{'ключ':<13}{'п.':<3}{'test acc, 3 seeds':<28}{'середнє':>9}{'train':>8}{'час':>7}")
    for key, item, label, config in WINE_EXPERIMENTS:
        seeds = [by_key[key][s] for s in SEEDS]
        test = [r["final_test_acc"] for r in seeds]
        train_acc = [r["final_train_acc"] for r in seeds]
        entry = {
            "key": key, "item": item, "label": label, "description": config.describe(),
            "config": seeds[0]["config"], "test_acc": test,
            "test_mean": float(np.mean(test)), "test_std": float(np.std(test)),
            "train_mean": float(np.mean(train_acc)),
            "seconds": float(np.mean([r["seconds"] for r in seeds])),
        }
        table.append(entry)
        print(f"{key:<13}{item:<3}{'  '.join(f'{a:.3f}' for a in test):<28}"
              f"{entry['test_mean']:>9.3f}{entry['train_mean']:>8.3f}{entry['seconds']:>6.1f}s")

        run = seeds[0]
        for name, values in run["history"].items():
            curves[f"{key}/{name}"] = values
        curves[f"{key}/snap_epoch"] = np.asarray(run["snapshots"]["epoch"])
        curves[f"{key}/test_pred"] = np.stack(run["snapshots"]["test_pred"])
        if config.features == 2:
            for layer in range(len(run["snapshots"]["weights"][0])):
                weights[f"{key}/W{layer}"] = np.stack(
                    [snap[layer][0] for snap in run["snapshots"]["weights"]]).astype(np.float32)
                weights[f"{key}/b{layer}"] = np.stack(
                    [snap[layer][1] for snap in run["snapshots"]["weights"]]).astype(np.float32)
            weights[f"{key}/mean"] = run["mean"]
            weights[f"{key}/std"] = run["std"]

    best_2d = max((e for e in table if e["config"]["features"] == 2),
                  key=lambda e: (round(e["test_mean"], 4), -e["test_std"]))
    best_all = max(table, key=lambda e: (round(e["test_mean"], 4), -e["test_std"]))
    print(f"\nнайкраща на 2 ознаках: {best_2d['key']} ({best_2d['test_mean']:.3f})")
    print(f"найкраща загалом:      {best_all['key']} ({best_all['test_mean']:.3f})")

    RESULTS.mkdir(exist_ok=True)
    meta = {
        "dataset": "sklearn.datasets.load_wine", "seeds": list(SEEDS),
        "train_size": len(split.y_train), "test_size": len(split.y_test),
        "feature_names": split.feature_names, "class_names": split.class_names,
        "best_2d": best_2d["key"], "best_all": best_all["key"], "experiments": table,
    }
    (RESULTS / "wine.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    np.savez_compressed(RESULTS / "wine_curves.npz", **curves,
                        X_train=split.X_train, X_test=split.X_test,
                        y_train=split.y_train, y_test=split.y_test)
    np.savez_compressed(RESULTS / "wine_weights.npz", **weights)


def analyse_cancer(split):
    X, names = split.X_train, split.feature_names
    kept, dropped, corr = classifier.drop_correlated(X, names, threshold=0.9)

    counts = np.bincount(np.concatenate([split.y_train, split.y_test]))
    print(f"рядків: {len(split.y_train) + len(split.y_test)} "
          f"(train {len(split.y_train)}, test {len(split.y_test)}), ознак: {X.shape[1]}")
    print("класи: " + ", ".join(f"{n} = {c}" for n, c in zip(split.class_names, counts)))
    upper = np.triu_indices_from(corr, k=1)
    strong = np.abs(corr[upper]) > 0.9
    print(f"пар з |r| > 0.9: {strong.sum()} із {strong.size}")
    order = np.argsort(-np.abs(corr[upper]))[:5]
    for i in order:
        a, b = upper[0][i], upper[1][i]
        print(f"   r({names[a]}, {names[b]}) = {corr[a, b]:.4f}")
    print(f"залишено {len(kept)} ознак, видалено {len(dropped)}:")
    for name, partner, r in dropped:
        print(f"   - {name:<24} (r = {r:+.3f} з «{partner}»)")
    return kept, dropped, corr


def train_cancer():
    split = classifier.load_split("breast_cancer")
    kept, dropped, corr = analyse_cancer(split)
    everything = list(range(split.X_train.shape[1]))
    variants = {key: kept if filtered else everything
                for key, (_, filtered, _) in CANCER_VARIANTS.items()}
    tasks = [(variant, columns, seed) for variant, columns in variants.items() for seed in SEEDS]
    runs = run_all(_cancer_job, tasks)

    summary, arrays = {}, {"corr": corr, "y_test": split.y_test}
    for variant, columns in variants.items():
        seeds = sorted((r for r in runs if r["key"] == variant), key=lambda r: r["seed"])
        metrics = []
        for run in seeds:
            pred = run["snapshots"]["test_pred"][-1]
            metrics.append(binary_metrics(split.y_test, pred))
        mean = {name: float(np.mean([m[name] for m in metrics])) for name in metrics[0]}
        run = seeds[0]
        label, _, config = CANCER_VARIANTS[variant]
        summary[variant] = {
            "label": label, "description": config.describe(), "epochs": config.epochs,
            "n_features": len(columns), "columns": columns,
            "test_acc": [m["accuracy"] for m in metrics], "mean": mean,
            "confusion": classifier.confusion_matrix(
                split.y_test, run["snapshots"]["test_pred"][-1], 2).tolist(),
            "seconds": float(np.mean([r["seconds"] for r in seeds])),
            "min_test_loss_epoch": int(run["history"]["epoch"][np.argmin(run["history"]["test_loss"])]),
            "min_test_loss": float(run["history"]["test_loss"].min()),
            "final_test_loss": float(run["history"]["test_loss"][-1]),
            "final_train_loss": float(run["history"]["train_loss"][-1]),
        }
        for name, values in run["history"].items():
            arrays[f"{variant}/{name}"] = values
        arrays[f"{variant}/snap_epoch"] = np.asarray(run["snapshots"]["epoch"])
        arrays[f"{variant}/test_pred"] = np.stack(run["snapshots"]["test_pred"])
        print(f"\n{variant}: {config.describe()}, епох {config.epochs}")
        print(f"{variant:<9} ознак {len(columns):>2}: acc по seed "
              + "  ".join(f"{a:.4f}" for a in summary[variant]["test_acc"])
              + f" | середнє acc {mean['accuracy']:.4f}, precision(M) {mean['precision']:.4f}, "
                f"recall(M) {mean['recall']:.4f}, F1(M) {mean['f1']:.4f}, "
                f"{summary[variant]['seconds']:.1f} с/запуск")
        print(f"          матриця помилок (seed 0, рядки — істина M/B): "
              f"{summary[variant]['confusion']}")
        print(f"          test loss: мінімум {summary[variant]['min_test_loss']:.4f} на епосі "
              f"{summary[variant]['min_test_loss_epoch']}, в кінці "
              f"{summary[variant]['final_test_loss']:.4f}; train loss в кінці "
              f"{summary[variant]['final_train_loss']:.4f}")

    meta = {
        "dataset": "sklearn.datasets.load_breast_cancer", "seeds": list(SEEDS),
        "train_size": len(split.y_train), "test_size": len(split.y_test),
        "feature_names": split.feature_names, "class_names": split.class_names,
        "kept": kept, "dropped": dropped, "variants": summary,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "cancer.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    np.savez_compressed(RESULTS / "cancer.npz", **arrays)


def binary_metrics(y_true, y_pred, positive=0):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = np.sum((y_pred == positive) & (y_true == positive))
    fp = np.sum((y_pred == positive) & (y_true != positive))
    fn = np.sum((y_pred != positive) & (y_true == positive))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {"accuracy": float(np.mean(y_true == y_pred)), "precision": float(precision),
            "recall": float(recall),
            "f1": float(2 * precision * recall / max(precision + recall, 1e-12))}


if __name__ == "__main__":
    parts = sys.argv[1:] or ["wine", "cancer"]
    if "wine" in parts:
        train_wine()
    if "cancer" in parts:
        print()
        train_cancer()
