import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision

import lenet

SEED = 18
LAB = Path(__file__).resolve().parents[1]
RESULTS = LAB / "results"
DATA = LAB.parent / ".data"
PROBE_SIZE = 48

MNIST_EPOCHS = 20
CIFAR_EPOCHS = 15

MNIST_RUNS = [
    ("tanh", "tanh, 5×5, avg", lambda: lenet.LeNet5(activation="tanh", conv_size=5), {}),
    ("relu", "relu, 5×5, avg", lambda: lenet.LeNet5(activation="relu", conv_size=5), {}),
    ("relu_3", "relu, 2×3×3, avg", lambda: lenet.LeNet5(activation="relu", conv_size=3), {}),
    ("relu_3_max_pool", "relu, 2×3×3, max",
     lambda: lenet.LeNet5(activation="relu", conv_size=3, pooling="max"), {}),
    ("relu_3_max_pool_bn", "relu, 2×3×3, max, BN",
     lambda: lenet.LeNet5(activation="relu", conv_size=3, pooling="max",
                          use_batch_norm=True), {}),
]


def cifar_lenet(**kwargs):
    return lenet.LeNet5(in_channels=3, image_size=32, **kwargs)


BEST_LENET = dict(activation="relu", conv_size=3, pooling="max", use_batch_norm=True)

CIFAR_RUNS = [
    *[(run_id, label, (lambda f=factory_kwargs: cifar_lenet(**f)), {})
      for run_id, label, factory_kwargs in [
          ("tanh", "tanh, 5×5, avg", dict(activation="tanh", conv_size=5)),
          ("relu", "relu, 5×5, avg", dict(activation="relu", conv_size=5)),
          ("relu_3", "relu, 2×3×3, avg", dict(activation="relu", conv_size=3)),
          ("relu_3_max_pool", "relu, 2×3×3, max",
           dict(activation="relu", conv_size=3, pooling="max")),
          ("relu_3_max_pool_bn", "relu, 2×3×3, max, BN", BEST_LENET),
      ]],
    ("wide", "+ фільтри 16/32", lambda: cifar_lenet(channels=(16, 32), **BEST_LENET), {}),
    ("wide_pad", "+ padding=1", lambda: cifar_lenet(channels=(16, 32), pad_all=True,
                                                     **BEST_LENET), {}),
    ("wide_pad_inbn", "+ BN входу", lambda: cifar_lenet(channels=(16, 32), pad_all=True,
                                                         input_batch_norm=True,
                                                         **BEST_LENET), {}),
    ("cifarnet", "CifarNet (BN)", lambda: lenet.CifarNet(), {}),
    ("cifarnet_nobn", "CifarNet без BN", lambda: lenet.CifarNet(batch_norm=False), {}),
    ("cifarnet_dropout", "CifarNet + dropout",
     lambda: lenet.CifarNet(dropout=0.3, conv_dropout=0.1), {}),
    *[(f"cifarnet_wd{exponent}", f"CifarNet, wd=1e-{exponent}", lambda: lenet.CifarNet(),
       {"weight_decay": 10.0 ** -exponent}) for exponent in (2, 3, 4, 5, 6)],
]


class Tee:

    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        print(text, flush=True)
        self.lines.append(text)


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def load_mnist(device):
    train = torchvision.datasets.MNIST(str(DATA), train=True, download=False)
    test = torchvision.datasets.MNIST(str(DATA), train=False, download=False)

    def tensors(dataset):
        X = dataset.data.float().unsqueeze(1) / 255.0
        return X.to(device), dataset.targets.long().to(device)

    return (*tensors(train), *tensors(test), [str(i) for i in range(10)])


def load_cifar10(device):
    train = torchvision.datasets.CIFAR10(str(DATA), train=True, download=False)
    test = torchvision.datasets.CIFAR10(str(DATA), train=False, download=False)

    def tensors(dataset):
        X = torch.from_numpy(dataset.data).permute(0, 3, 1, 2).float() / 255.0
        return X.to(device), torch.tensor(dataset.targets).long().to(device)

    return (*tensors(train), *tensors(test), list(train.classes))


def run_networks(part, runs, loader, epochs, only=None):
    device = pick_device()
    torch.set_num_threads(4)
    log = Tee()
    X_train, y_train, X_test, y_test, classes = loader(device)
    log(f"{part}: train {tuple(X_train.shape)}, test {tuple(X_test.shape)}, device {device}, "
        f"epochs {epochs}, seed {SEED}")

    generator = np.random.default_rng(SEED)
    probe_index = np.sort(generator.choice(len(X_test), PROBE_SIZE, replace=False))
    probe = X_test[torch.from_numpy(probe_index).to(device)]

    json_path, npz_path = RESULTS / f"{part}.json", RESULTS / f"{part}.npz"
    summary = json.loads(json_path.read_text()) if only and json_path.exists() else {}
    arrays = dict(np.load(npz_path)) if only and npz_path.exists() else {}
    summary.update({"dataset": part, "epochs": epochs, "seed": SEED, "device": str(device),
                    "classes": classes, "probe_index": probe_index.tolist()})
    summary.setdefault("runs", {})
    arrays["probe_images"] = (probe.cpu().numpy() * 255).round().astype(np.uint8)
    arrays["probe_labels"] = y_test[torch.from_numpy(probe_index).to(device)].cpu().numpy()
    arrays["test_labels"] = y_test.cpu().numpy().astype(np.uint8)

    for run_id, label, factory, optimiser in runs:
        if only and run_id not in only:
            continue
        torch.manual_seed(SEED)
        net = factory()
        log(f"\n[{run_id}] {label}: {lenet.count_parameters(net)} parameters, {optimiser or ''}")
        started = time.monotonic()
        history = lenet.train(net, X_train, y_train, X_test, y_test, epochs,
                              probe=probe, device=device, log=log, **optimiser)
        elapsed = time.monotonic() - started
        best = int(np.argmax(history["test_acc"]))
        log(f"  best test acc {history['test_acc'][best]:.4f} at epoch {best + 1}, "
            f"min test loss at epoch {int(np.argmin(history['test_loss'])) + 1}, "
            f"{elapsed:.0f}s")

        summary["runs"][run_id] = {
            "label": label,
            "parameters": lenet.count_parameters(net),
            "optimiser": optimiser,
            "seconds": round(elapsed, 1),
            **{key: [round(v, 5) for v in history[key]]
               for key in ("train_loss", "train_acc", "test_loss", "test_acc")},
        }
        arrays[f"probe_{run_id}"] = np.stack(history["probe"]).astype(np.float16)
        arrays[f"predictions_{run_id}"] = history["predictions"].astype(np.uint8)

        order = [run[0] for run in runs if run[0] in summary["runs"]]
        summary["runs"] = {key: summary["runs"][key] for key in order}
        json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=1))
        np.savez_compressed(npz_path, **arrays)
        log_name = f"{part}-rerun.log" if only else f"{part}.log"
        (RESULTS / log_name).write_text("\n".join(log.lines) + "\n", encoding="utf-8")


def run_gradients():
    log = Tee()
    log(f"SimpleNet: 3 × Linear(1, 1, bias=False), weights = 1, "
        f"{lenet.NUMBER_OF_EXPERIMENTS} inputs ~ N(0, 1), seed {SEED}")
    means = lenet.gradient_probe(SEED)
    seeds = [SEED + k for k in range(10)]
    per_seed = {seed: lenet.gradient_probe(seed) for seed in seeds}
    for name, value in sorted(means.items(), key=lambda item: -item[1]):
        spread = [per_seed[seed][name] for seed in seeds]
        log(f"  {name:<11} {value:.6f}   (seeds {seeds[0]}-{seeds[-1]}: "
            f"{min(spread):.4f} .. {max(spread):.4f})")
    winners = {seed: max(values, key=values.get) for seed, values in per_seed.items()}
    log(f"  winner per seed: {winners}")
    (RESULTS / "gradients.json").write_text(json.dumps(
        {"seed": SEED, "experiments": lenet.NUMBER_OF_EXPERIMENTS, "mean_grad": means,
         "per_seed": {str(k): v for k, v in per_seed.items()}},
        ensure_ascii=False, indent=1))
    (RESULTS / "gradients.log").write_text("\n".join(log.lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("part", choices=("gradients", "mnist", "cifar10"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--only", nargs="*", help="rerun just these run ids, keep the rest")
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)

    if args.part == "gradients":
        run_gradients()
    elif args.part == "mnist":
        run_networks("mnist", MNIST_RUNS, load_mnist, args.epochs or MNIST_EPOCHS, args.only)
    else:
        run_networks("cifar10", CIFAR_RUNS, load_cifar10, args.epochs or CIFAR_EPOCHS,
                     args.only)


if __name__ == "__main__":
    sys.exit(main())
