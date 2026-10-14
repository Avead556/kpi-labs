import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import builders

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
DATA_ROOT = HERE.parents[1] / ".data"

SEED = 18
BATCH_SIZE = 100
LEARNING_RATE = 1.0e-3
EPOCHS = {"mnist": 10, "cifar": 8}
SAMPLES = 32


class Tee:

    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")

    def __call__(self, *parts):
        line = " ".join(str(part) for part in parts)
        print(line, flush=True)
        self.file.write(line + "\n")
        self.file.flush()


def load(dataset):
    from torchvision import datasets

    def tensors(train):
        if dataset == "mnist":
            source = datasets.MNIST(DATA_ROOT, train=train, download=False)
            images = source.data.float().unsqueeze(1) / 255.0
            labels = source.targets.clone()
        else:
            source = datasets.CIFAR10(DATA_ROOT, train=train, download=False)
            images = torch.from_numpy(source.data).float().permute(0, 3, 1, 2) / 255.0
            labels = torch.tensor(source.targets)
        return images, labels

    return (*tensors(True), *tensors(False))


def evaluate(net, images, labels, loss_fn, batch=1000):
    net.eval()
    total_loss, correct, logits_all = 0.0, 0, []
    with torch.no_grad():
        for start in range(0, len(images), batch):
            logits = net(images[start:start + batch])
            target = labels[start:start + batch]
            total_loss += loss_fn(logits, target).item() * len(target)
            correct += (logits.argmax(1) == target).sum().item()
            logits_all.append(logits)
    return total_loss / len(images), correct / len(images), torch.cat(logits_all)


def train(net, data, epochs, device, log):
    X_train, y_train, X_test, y_test = data
    net = net.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    order_rng = np.random.default_rng(SEED)
    history = {"train_loss": [], "test_loss": [], "test_acc": [], "seconds": []}

    for epoch in range(epochs):
        started = time.perf_counter()
        order = torch.from_numpy(order_rng.permutation(len(X_train))).to(device)
        net.train()
        running, batches = 0.0, 0
        for start in range(0, len(X_train), BATCH_SIZE):
            index = order[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            loss_value = loss_fn(net(X_train[index]), y_train[index])
            loss_value.backward()
            optimizer.step()
            running += loss_value.item()
            batches += 1

        test_loss, test_acc, _ = evaluate(net, X_test, y_test, loss_fn)
        seconds = time.perf_counter() - started
        history["train_loss"].append(running / batches)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["seconds"].append(seconds)
        log(f"  epoch {epoch + 1:2d}  train loss {running / batches:.4f}  "
            f"test loss {test_loss:.4f}  test acc {test_acc:.4f}  {seconds:.1f}s")
    return net, history


def describe(model):
    children = [{"name": name, "type": type(child).__name__,
                 "params": builders.count_parameters(child)}
                for name, child in model.named_children()]
    kinds = {}
    for module in model.modules():
        if not list(module.children()):
            kinds[type(module).__name__] = kinds.get(type(module).__name__, 0) + 1
    groups = {}
    for name, parameter in model.named_parameters():
        parts = name.split(".")[:-1]
        depth = 3 if len(parts) >= 3 and parts[1] == "blocks" else 2
        group = ".".join(parts[:depth])
        groups[group] = groups.get(group, 0) + parameter.numel()
    containers = sorted({type(module).__name__ for module in model.modules()
                         if isinstance(module, (nn.Sequential, nn.ModuleList,
                                                nn.ModuleDict))})
    return {
        "repr": repr(model),
        "type": type(model).__name__,
        "params": builders.count_parameters(model),
        "modules": sum(1 for _ in model.modules()),
        "leaves": sum(kinds.values()),
        "children": children,
        "groups": [[name, count] for name, count in groups.items()],
        "kinds": kinds,
        "containers": containers,
        "state_keys": list(model.state_dict().keys())[:6],
    }


def check_equivalence(classic, dataset, data, device, log):
    _, _, X_test, y_test = data
    loss_fn = nn.CrossEntropyLoss()
    _, reference_acc, reference = evaluate(classic, X_test, y_test, loss_fn)
    report = {"classic_acc": reference_acc}
    for key in ("sequential", "modular"):
        other = builders.build(dataset, key).to(device)
        builders.copy_weights(classic, other)
        _, acc, logits = evaluate(other, X_test, y_test, loss_fn)
        diff = (logits - reference).abs().max().item()
        same = (logits.argmax(1) == reference.argmax(1)).float().mean().item()
        report[key] = {"acc": acc, "max_logit_diff": diff, "same_argmax": same}
        log(f"  classic -> {key}: acc {acc:.4f} (classic {reference_acc:.4f}), "
            f"max |logit diff| {diff:.2e}, same predictions {same:.4f}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=list(builders.MODELS))
    parser.add_argument("--cpu", action="store_true", help="do not use MPS / CUDA")
    parser.add_argument("--describe-only", action="store_true",
                        help="only rewrite models.json, no training")
    args = parser.parse_args()

    if args.describe_only:
        RESULTS.mkdir(exist_ok=True)
        models = {dataset: {} for dataset in builders.MODELS}
        for dataset, entries in builders.MODELS.items():
            for key, (label, _) in entries.items():
                torch.manual_seed(SEED)
                models[dataset][key] = {"label": label,
                                        **describe(builders.build(dataset, key))}
        (RESULTS / "models.json").write_text(json.dumps(models, ensure_ascii=False, indent=1))
        return

    torch.set_num_threads(4)
    if args.cpu:
        device = torch.device("cpu")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    RESULTS.mkdir(exist_ok=True)
    log = Tee(RESULTS / ("train_log.txt" if not args.only else f"train_log_{args.only}.txt"))
    log(f"torch {torch.__version__}, device {device}, seed {SEED}, "
        f"batch {BATCH_SIZE}, Adam lr {LEARNING_RATE}")

    history_path, models_path = RESULTS / "history.json", RESULTS / "models.json"
    equivalence_path = RESULTS / "equivalence.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else {}
    models = json.loads(models_path.read_text()) if models_path.exists() else {}
    equivalence = json.loads(equivalence_path.read_text()) if equivalence_path.exists() else {}
    samples = dict(np.load(RESULTS / "samples.npz")) if (RESULTS / "samples.npz").exists() else {}

    datasets = [args.only] if args.only else list(builders.MODELS)
    for dataset in datasets:
        started = time.perf_counter()
        data = [tensor.to(device) for tensor in load(dataset)]
        log(f"\n== {dataset}: train {tuple(data[0].shape)}, test {tuple(data[2].shape)}, "
            f"loaded in {time.perf_counter() - started:.1f}s")
        history[dataset], models[dataset] = {}, {}
        trained = {}

        for key, (label, _) in builders.MODELS[dataset].items():
            torch.manual_seed(SEED)
            model = builders.build(dataset, key)
            models[dataset][key] = {"label": label, **describe(model)}
            log(f"\n-- {dataset}/{key} ({label}): {models[dataset][key]['params']} parameters")
            model, run = train(model, data, EPOCHS[dataset], device, log)
            history[dataset][key] = {"label": label, **run}
            trained[key] = model

        log(f"\n-- {dataset}: classic weights in the other styles")
        equivalence[dataset] = check_equivalence(trained["classic"], dataset, data, device, log)

        modular = trained["modular"].cpu().eval()
        torch.save(modular.state_dict(), RESULTS / f"{dataset}_modular.pt")
        samples[f"{dataset}_images"] = data[2][:SAMPLES].cpu().numpy().astype(np.float32)
        samples[f"{dataset}_labels"] = data[3][:SAMPLES].cpu().numpy().astype(np.int64)

        history_path.write_text(json.dumps(history, ensure_ascii=False, indent=1))
        models_path.write_text(json.dumps(models, ensure_ascii=False, indent=1))
        equivalence_path.write_text(json.dumps(equivalence, indent=1))
        np.savez_compressed(RESULTS / "samples.npz", **samples)

    log("\nsummary (best / last test accuracy, mean epoch time):")
    for dataset in datasets:
        for key, run in history[dataset].items():
            log(f"  {dataset:6s} {key:14s} {max(run['test_acc']):.4f} / {run['test_acc'][-1]:.4f}"
                f"  {np.mean(run['seconds']):.1f}s")


if __name__ == "__main__":
    sys.exit(main())
