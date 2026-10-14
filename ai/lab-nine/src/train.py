import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import lenet

RESULTS = Path(__file__).resolve().parents[1] / "results"
HISTORY = RESULTS / "history.json"
LOG = RESULTS / "train_log.txt"

LENET_EPOCHS = 20
PLUS_EPOCHS = 10
CIFAR_EPOCHS = 25
VARIANCE_SEEDS = [0, 1, 2, 3, 4]
VARIANCE_EPOCHS = {"lenet5": 5, "lenet5plus": 3}
SAMPLES = 500
STAGES = ("lenet", "plus", "variance", "timing", "cifar", "padding")


def log(message=""):
    print(message, flush=True)
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(message + "\n")


def load_results():
    return json.loads(HISTORY.read_text()) if HISTORY.exists() else {}


def save_results(results):
    HISTORY.write_text(json.dumps(results, indent=1, ensure_ascii=False))


def confusion(model, X, y, device):
    predictions = []
    with torch.no_grad():
        model.eval()
        for start in range(0, len(X), lenet.EVAL_CHUNK):
            predictions.append(model(X[start:start + lenet.EVAL_CHUNK].to(device)).argmax(1).cpu())
    predicted = torch.cat(predictions).numpy()
    matrix = np.zeros((10, 10), dtype=int)
    np.add.at(matrix, (y.numpy(), predicted), 1)
    return matrix.tolist()


def run_model(name, factory, data, epochs, device, weights):
    torch.manual_seed(0)
    model = factory()
    log(f"\n== {name}: {lenet.count_parameters(model)} parameters, {epochs} epochs on {device}")
    started = time.perf_counter()
    history = lenet.train(model, data, epochs, device, seed=0, log=log)
    total = time.perf_counter() - started
    torch.save({key: value.cpu() for key, value in model.state_dict().items()}, RESULTS / weights)
    best = int(np.argmax(history["test_acc"]))
    log(f"   best test acc {history['test_acc'][best]:.4f} at epoch {best + 1}; "
        f"final {history['test_acc'][-1]:.4f}; {total:.0f} s total")
    return {**history, "params": lenet.count_parameters(model), "epochs": epochs,
            "total_seconds": total, "weights": weights,
            "confusion": confusion(model, data[2], data[3], device)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=STAGES, default=list(STAGES))
    args = parser.parse_args()

    torch.set_num_threads(4)
    RESULTS.mkdir(exist_ok=True)
    device = lenet.best_device()
    results = load_results()
    results["device"] = str(device)
    results["torch"] = torch.__version__
    log(f"\n######## {time.strftime('%Y-%m-%d %H:%M')}  stages: {', '.join(args.only)}  "
        f"device: {device}  torch {torch.__version__}")

    mnist = None
    if {"lenet", "plus", "variance"} & set(args.only):
        mnist = lenet.load_mnist()
        log(f"MNIST: train {tuple(mnist[0].shape)}, test {tuple(mnist[2].shape)}")

    if "lenet" in args.only:
        results["lenet5"] = run_model("LeNet5 / MNIST", lenet.LeNet5, mnist,
                                      LENET_EPOCHS, device, "lenet5_mnist.pt")
        save_results(results)

    if "plus" in args.only:
        results["lenet5plus"] = run_model("LeNet5Plus / MNIST", lenet.LeNet5Plus, mnist,
                                          PLUS_EPOCHS, device, "lenet5plus_mnist.pt")
        save_results(results)

    if "variance" in args.only:
        variance = {}
        for key, factory in (("lenet5", lenet.LeNet5), ("lenet5plus", lenet.LeNet5Plus)):
            epochs = VARIANCE_EPOCHS[key]
            best = []
            for seed in VARIANCE_SEEDS:
                log(f"\n== variance {key}, seed {seed}, {epochs} epochs")
                torch.manual_seed(seed)
                history = lenet.train(factory(), mnist, epochs, device, seed=seed, log=log)
                best.append(max(history["test_acc"]))
            best = np.array(best)
            variance[key] = {"epochs": epochs, "seeds": VARIANCE_SEEDS, "best": best.tolist(),
                             "mean": best.mean(), "std": best.std(ddof=1),
                             "spread": best.max() - best.min()}
            log(f"   {key}: best acc per seed {np.round(best, 4).tolist()}, "
                f"mean {best.mean():.4f} ± {best.std(ddof=1):.4f}")
        pooled = np.sqrt((variance["lenet5"]["std"] ** 2 + variance["lenet5plus"]["std"] ** 2) / 2)
        needed = {f"{delta:g}": lenet.runs_needed(pooled, delta) for delta in (0.0005, 0.001, 0.002)}
        variance["pooled_std"] = pooled
        variance["runs_needed"] = needed
        log(f"   pooled std {pooled:.5f}; runs per architecture to detect a difference of "
            + ", ".join(f"{delta}: {n}" for delta, n in needed.items()))
        results["variance"] = variance
        save_results(results)

    if "timing" in args.only:
        timing = {}
        for device_name in dict.fromkeys([device.type, "cpu"]):
            for key, factory in (("lenet5", lenet.LeNet5), ("lenet5plus", lenet.LeNet5Plus)):
                rows = lenet.layer_timings(factory(), (lenet.BATCH_SIZE, 1, 28, 28),
                                           torch.device(device_name))
                timing[f"{key}/{device_name}"] = rows
                log(f"\n== layer time, {key} on {device_name}, batch {lenet.BATCH_SIZE}, "
                    "forward + backward")
                for row in rows:
                    log(f"   {row['layer']:<14} {row['kind']:<12} in {str(row['input']):<14} "
                        f"{row['params']:>7} w  {row['ms']:7.3f} ms")
        results["timing"] = timing
        save_results(results)

    if "cifar" in args.only:
        cifar = lenet.load_cifar10()
        log(f"\nCIFAR-10: train {tuple(cifar[0].shape)}, test {tuple(cifar[2].shape)}")
        results["cifar"] = run_model("LeNet5 / CIFAR-10 (in_channels=3, padding=0)",
                                     lambda: lenet.LeNet5(in_channels=3, padding=0), cifar,
                                     CIFAR_EPOCHS, device, "lenet5_cifar.pt")
        best_loss = int(np.argmin(results["cifar"]["test_loss"]))
        log(f"   minimum test loss {results['cifar']['test_loss'][best_loss]:.4f} "
            f"at epoch {best_loss + 1}")
        save_results(results)
        _save_samples(cifar=cifar)

    if "padding" in args.only:
        rows = lenet.padding_exercise()
        log("\n== task 4: padding for x of shape (4, 3, 8, 16)")
        for row in rows:
            log(f"   kernel_size={row['kernel']}, padding={row['padding']} -> "
                f"torch.Size({list(row['shape'])})")
        results["padding"] = rows
        save_results(results)

    if mnist is not None:
        _save_samples(mnist=mnist)
    log("\ndone")


def _save_samples(mnist=None, cifar=None):
    path = RESULTS / "samples.npz"
    stored = dict(np.load(path)) if path.exists() else {}
    if mnist is not None:
        stored["mnist_images"] = mnist[2][:SAMPLES, 0].numpy().astype(np.uint8)
        stored["mnist_labels"] = mnist[3][:SAMPLES].numpy()
    if cifar is not None:
        stored["cifar_images"] = cifar[2][:SAMPLES].numpy().astype(np.uint8)
        stored["cifar_labels"] = cifar[3][:SAMPLES].numpy()
    np.savez_compressed(path, **stored)


if __name__ == "__main__":
    sys.exit(main())
