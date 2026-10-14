import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import fcnet

RESULTS = Path(__file__).resolve().parents[1] / "results"
SEEDS = (0, 1, 2, 3, 4)
OPTIMIZER_EPOCHS = 20
MNIST_ABLATION_EPOCHS = 40
MNIST_LONG_EPOCHS = 200
CIFAR_EPOCHS = 40
IMPROVED_MNIST = {"hidden": (256,), "activation": "relu"}
IMPROVED_CIFAR = {"hidden": (512, 256), "activation": "relu"}


def log(message=""):
    print(message, flush=True)


def save(name, payload):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / name).write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    log(f"-> results/{name}")


def rounded(history):
    return {key: [round(value, 5) for value in values] for key, values in history.items()}


def run(data, epochs, seed=0, optimizer="Adam", keep_best=False, quiet=False, **net_kw):
    fcnet.seed_everything(seed)
    net = fcnet.FCNet(data.features, **net_kw)
    history, best = fcnet.train(net, data, fcnet.make_optimizer(optimizer, net.parameters()),
                                epochs, keep_best=keep_best, log=None if quiet else log)
    return net, history, best


def section_linear():
    log("== 1g: fully connected layer 2 -> 3 by matrix multiplication")
    result = fcnet.linear_by_matmul()
    log(f"fc_out:              {result['fc_out'].tolist()}")
    log(f"fc_out_alternative:  {result['fc_out_alternative'].tolist()}")
    log(f"fc_out == alternative: {result['equal'].tolist()}")
    for key in ("fc_weight_grad", "our_weight_grad", "fc_bias_grad", "our_bias_grad"):
        log(f"{key + ':':<17} {result[key].tolist()}")

    log("\n== 1e: ScratchLinear against torch.nn.Linear")
    torch.manual_seed(0)
    original = torch.nn.Linear(784, 256)
    torch.manual_seed(0)
    scratch = fcnet.ScratchLinear(784, 256)
    init_diff = (original.weight - scratch.weight).abs().max().item()
    init_bias_diff = (original.bias - scratch.bias).abs().max().item()

    x = torch.randn(64, 784)
    target = torch.randn(64, 256)
    for layer in (original, scratch):
        ((layer(x) - target) ** 2).mean().backward()
    out_diff = (original(x) - scratch(x)).abs().max().item()
    weight_grad_diff = (original.weight.grad - scratch.weight.grad).abs().max().item()
    bias_grad_diff = (original.bias.grad - scratch.bias.grad).abs().max().item()
    log(f"same seed, max |W - W'| after init:  {init_diff:.3e}  (bias {init_bias_diff:.3e})")
    log(f"max |y - y'| on a batch of 64:       {out_diff:.3e}")
    log(f"max |dL/dW - dL/dW'|:                 {weight_grad_diff:.3e}")
    log(f"max |dL/db - dL/db'|:                 {bias_grad_diff:.3e}")

    log("\n   whole MNIST training, 5 epochs, nn.Linear vs ScratchLinear:")
    data = fcnet.load_mnist()
    finals = {}
    for name, linear in (("nn.Linear", torch.nn.Linear), ("ScratchLinear", fcnet.ScratchLinear)):
        fcnet.seed_everything(0)
        net = fcnet.FCNet(784, linear=linear, **IMPROVED_MNIST)
        history, _ = fcnet.train(net, data, torch.optim.Adam(net.parameters(), 1e-3), 5)
        finals[name] = {"test_acc": history["test_acc"], "test_loss": history["test_loss"],
                        "seconds": sum(history["seconds"])}
        log(f"   {name:<14} test acc by epoch {[round(a, 4) for a in history['test_acc']]}"
            f"  ({finals[name]['seconds']:.1f} s)")

    save("linear.json", {
        "matmul": {key: value.tolist() for key, value in result.items()},
        "scratch": {"init_weight_diff": init_diff, "init_bias_diff": init_bias_diff,
                    "output_diff": out_diff, "weight_grad_diff": weight_grad_diff,
                    "bias_grad_diff": bias_grad_diff, "training": finals},
    })


def section_mnist():
    raw = fcnet.load_mnist(normalize=False)
    data = fcnet.load_mnist()
    log(f"MNIST: train {tuple(data.x_train.shape)}, val {tuple(data.x_val.shape)}, "
        f"test {tuple(data.x_test.shape)}, dtype {data.x_train.dtype}")

    steps = [
        ("manual", "як у посібнику: 0..255, сигмоїда, 100", raw, {"hidden": (100,), "activation": "sigmoid"}),
        ("normalized", "+ нормування /255", data, {"hidden": (100,), "activation": "sigmoid"}),
        ("relu", "+ ReLU замість сигмоїди", data, {"hidden": (100,), "activation": "relu"}),
    ]
    runs = {}
    for key, label, split, net_kw in steps:
        log(f"\n== MNIST {key}: {label}, {MNIST_ABLATION_EPOCHS} epochs")
        _, history, _ = run(split, MNIST_ABLATION_EPOCHS, **net_kw)
        runs[key] = {"label": label, "net": net_kw, "history": rounded(history)}

    log(f"\n== MNIST improved: ReLU, 256 hidden, {MNIST_LONG_EPOCHS} epochs")
    net, history, best = run(data, MNIST_LONG_EPOCHS, keep_best=True, **IMPROVED_MNIST)
    runs["improved"] = {"label": "+ 256 нейронів, 200 епох", "net": IMPROVED_MNIST,
                        "history": rounded(history)}

    net.load_state_dict(best["state"])
    loss_fn = torch.nn.CrossEntropyLoss()
    test_loss, test_acc = fcnet.evaluate(net, data.x_test, data.y_test, loss_fn)
    with torch.no_grad():
        predicted = net(data.x_test).argmax(dim=1)
    confusion = np.zeros((10, 10), dtype=int)
    np.add.at(confusion, (data.y_test.numpy(), predicted.numpy()), 1)
    at40 = history["test_acc"][39]
    log(f"\nbest validation loss at epoch {best['epoch']} ({best['val_loss']:.4f}); "
        f"test accuracy there {test_acc:.4f}, test loss {test_loss:.4f}")
    log(f"test accuracy after 40 epochs {at40:.4f}, after {MNIST_LONG_EPOCHS} "
        f"{history['test_acc'][-1]:.4f}")

    linear_layers = [m for m in net.layers if isinstance(m, torch.nn.Linear)]
    np.savez_compressed(
        RESULTS / "mnist_mlp.npz",
        w1=linear_layers[0].weight.detach().numpy(), b1=linear_layers[0].bias.detach().numpy(),
        w2=linear_layers[1].weight.detach().numpy(), b2=linear_layers[1].bias.detach().numpy(),
        epoch=best["epoch"])
    log("-> results/mnist_mlp.npz")
    save("mnist.json", {
        "runs": runs, "best_epoch": best["epoch"], "best_val_loss": best["val_loss"],
        "best_test_acc": test_acc, "best_test_loss": test_loss,
        "confusion": confusion.tolist(),
        "errors": np.flatnonzero(predicted.numpy() != data.y_test.numpy()).tolist(),
    })


def section_determinism():
    data = fcnet.load_mnist()
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    results = []
    for device in devices:
        moved = fcnet.Split.__new__(fcnet.Split)
        for key, value in vars(data).items():
            setattr(moved, key, value.to(device) if torch.is_tensor(value) else value)
        for repeat in range(2):
            for flag in (True, False):
                fcnet.seed_everything(0)
                torch.backends.cudnn.deterministic = flag
                net = fcnet.FCNet(784, **IMPROVED_MNIST).to(device)
                history, _ = fcnet.train(net, moved, torch.optim.Adam(net.parameters(), 1e-3), 3)
                seconds = sum(history["seconds"])
                results.append({"device": device, "deterministic": flag, "repeat": repeat,
                                "seconds": seconds, "test_acc": history["test_acc"][-1]})
                log(f"{device:<4} deterministic={flag!s:<5} repeat {repeat}: "
                    f"{seconds:.2f} s for 3 epochs, test acc {history['test_acc'][-1]:.4f}")
    torch.backends.cudnn.deterministic = True
    save("determinism.json", {"cuda": torch.cuda.is_available(), "runs": results})


def section_optimizers():
    data = fcnet.load_mnist()
    results = {}
    for name, (_, lr) in fcnet.OPTIMIZERS.items():
        results[name] = {"lr": lr, "seeds": {}}
        for seed in SEEDS:
            _, history, _ = run(data, OPTIMIZER_EPOCHS, seed=seed, optimizer=name,
                                quiet=True, **IMPROVED_MNIST)
            results[name]["seeds"][seed] = rounded(history)
            log(f"{name:<9} lr={lr:<6} seed {seed}: val acc {history['val_acc'][-1]:.4f}  "
                f"best val loss {min(history['val_loss']):.4f} "
                f"@ {int(np.argmin(history['val_loss'])) + 1}  "
                f"test acc {history['test_acc'][-1]:.4f}  ({sum(history['seconds']):.1f} s)")
        finals = [results[name]["seeds"][s]["test_acc"][-1] for s in SEEDS]
        log(f"   {name}: test acc after {OPTIMIZER_EPOCHS} epochs "
            f"{np.mean(finals):.4f} ± {np.std(finals, ddof=1):.4f} "
            f"(min {min(finals):.4f}, max {max(finals):.4f})\n")
    save("optimizers.json", {"epochs": OPTIMIZER_EPOCHS, "seeds": list(SEEDS),
                             "net": IMPROVED_MNIST, "optimizers": results})


def section_cifar():
    started = time.perf_counter()
    data = fcnet.load_cifar10()
    log(f"CIFAR-10: train {tuple(data.x_train.shape)}, val {tuple(data.x_val.shape)}, "
        f"test {tuple(data.x_test.shape)}, loaded in {time.perf_counter() - started:.1f} s")
    log(f"classes: {data.classes}")

    configs = [
        ("manual", "сигмоїда, 100, Adam", {"hidden": (100,), "activation": "sigmoid"}, "Adam"),
        ("relu_adam", "ReLU, 512-256, Adam", IMPROVED_CIFAR, "Adam"),
        ("relu_momentum", "ReLU, 512-256, SGD+momentum", IMPROVED_CIFAR, "Momentum"),
        ("relu_rmsprop", "ReLU, 512-256, RMSprop", IMPROVED_CIFAR, "RMSprop"),
    ]
    runs, best_run = {}, None
    for key, label, net_kw, optimizer in configs:
        log(f"\n== CIFAR-10 {key}: {label}, {CIFAR_EPOCHS} epochs")
        net, history, best = run(data, CIFAR_EPOCHS, optimizer=optimizer, keep_best=True, **net_kw)
        net.load_state_dict(best["state"])
        test_loss, test_acc = fcnet.evaluate(net, data.x_test, data.y_test,
                                             torch.nn.CrossEntropyLoss())
        runs[key] = {"label": label, "net": net_kw, "optimizer": optimizer,
                     "history": rounded(history), "best_epoch": best["epoch"],
                     "best_test_acc": test_acc, "best_test_loss": test_loss}
        log(f"best validation loss at epoch {best['epoch']} ({best['val_loss']:.4f}); "
            f"test accuracy there {test_acc:.4f}")
        if best_run is None or test_acc > runs[best_run[0]]["best_test_acc"]:
            best_run = (key, net)

    key, net = best_run
    with torch.no_grad():
        predicted = net(data.x_test).argmax(dim=1).numpy()
    truth = data.y_test.numpy()
    per_class = {name: float((predicted[truth == c] == c).mean())
                 for c, name in enumerate(data.classes)}
    log(f"\nper-class test accuracy of {key}:")
    for name, accuracy in per_class.items():
        log(f"   {name:<11} {accuracy:.3f}")
    save("cifar.json", {"classes": data.classes, "runs": runs,
                        "best": key, "per_class": per_class})


SECTIONS = {
    "linear": section_linear,
    "mnist": section_mnist,
    "determinism": section_determinism,
    "optimizers": section_optimizers,
    "cifar": section_cifar,
}


def main(names):
    torch.set_num_threads(4)
    RESULTS.mkdir(exist_ok=True)
    log(f"torch {torch.__version__}, cuda {torch.cuda.is_available()}, "
        f"mps {torch.backends.mps.is_available()}, threads {torch.get_num_threads()}")
    for name in names or SECTIONS:
        started = time.perf_counter()
        log(f"\n######## {name}")
        SECTIONS[name]()
        log(f"######## {name} done in {time.perf_counter() - started:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1:])
