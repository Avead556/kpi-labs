import json
from pathlib import Path

import numpy as np

from labkit import imagekit

RESULTS = Path(__file__).resolve().parents[1] / "results"
MNIST_RAW = Path(__file__).resolve().parents[2] / ".data" / "MNIST" / "raw"
FIGURE_SIZE = (1200, 780)
METRICS = (
    ("val_acc", "точність на валідації"),
    ("val_loss", "втрати на валідації"),
    ("train_loss", "втрати на навчальній вибірці"),
)
DIM_ALPHA = 0.18
ERROR_HEX = "#D8735F"


def _load(name):
    return json.loads((RESULTS / name).read_text())


def _int_param(params, key, default, low, high):
    try:
        value = int(round(float((params or {}).get(key, default))))
    except (TypeError, ValueError):
        value = default
    return max(low, min(value, high))


class _PlotScene:

    def __init__(self):
        self.quad = imagekit.ImageQuad()
        self._key = None

    def draw(self, width, height, params=None):
        key = self.key(params or {})
        if key != self._key:
            with imagekit.style():
                figure = self.figure(*key)
                self.quad.upload(imagekit.figure_to_rgb(figure))
            self._key = key
        self.quad.draw(width, height)


class OptimizersScene(_PlotScene):

    def __init__(self, options=None):
        super().__init__()
        data = _load("optimizers.json")
        self.epochs = data["epochs"]
        self.seeds = data["seeds"]
        self.names = list(data["optimizers"])
        self.lr = {name: data["optimizers"][name]["lr"] for name in self.names}
        self.curves = {
            name: {key: np.array([run[key] for run in data["optimizers"][name]["seeds"].values()])
                   for key in ("train_loss", "val_loss", "val_acc", "test_acc")}
            for name in self.names}

    def metadata(self):
        return {"description": "Методи оптимізації на MNIST, 5 random seed",
                "optimizers": self.names, "epochs": self.epochs}

    def key(self, params):
        return (_int_param(params, "epoch", self.epochs, 1, self.epochs),
                _int_param(params, "focus", 0, 0, len(self.names)),
                _int_param(params, "metric", 0, 0, len(METRICS) - 1))

    def figure(self, epoch, focus, metric):
        figure, (curve_axes, seed_axes) = imagekit.new_figure(
            *FIGURE_SIZE, ncols=2, gridspec_kw={"width_ratios": (1.55, 1.0)})
        metric_key, metric_label = METRICS[metric]
        epochs = np.arange(1, self.epochs + 1)

        for index, name in enumerate(self.names):
            color = imagekit.SERIES_HEX[index % len(imagekit.SERIES_HEX)]
            alpha = 1.0 if focus in (0, index + 1) else DIM_ALPHA
            values = self.curves[name][metric_key]
            if metric_key == "val_acc":
                values = values * 100
            curve_axes.fill_between(epochs, values.min(axis=0), values.max(axis=0),
                                    color=color, alpha=0.18 * alpha, linewidth=0)
            curve_axes.plot(epochs, values.mean(axis=0), color=color, alpha=alpha,
                            linewidth=2.0, label=name)

            accuracies = self.curves[name]["test_acc"][:, epoch - 1] * 100
            row = len(self.names) - 1 - index
            seed_axes.scatter(accuracies, np.full(len(accuracies), row), s=34,
                              color=color, alpha=alpha, zorder=3)
            seed_axes.plot([accuracies.mean()] * 2, [row - 0.3, row + 0.3],
                           color=color, alpha=alpha, linewidth=2.2)

        curve_axes.axvline(epoch, color=imagekit.MUTED_HEX, linewidth=1.0, linestyle="--")
        curve_axes.set_xlabel("епоха")
        curve_axes.set_title(f"{metric_label}: середнє і розкид за {len(self.seeds)} seed")
        curve_axes.set_xlim(1, self.epochs)
        curve_axes.set_xticks([1, 5, 10, 15, 20] if self.epochs == 20 else None)
        if metric_key == "val_acc":
            curve_axes.set_ylim(94, 98.8)
            curve_axes.set_ylabel("%")
        else:
            curve_axes.set_yscale("log")
        curve_axes.legend(loc="lower right" if metric_key == "val_acc" else "upper right",
                          fontsize=9, ncol=2)

        seed_axes.set_yticks(range(len(self.names)))
        seed_axes.set_yticklabels([f"{name}\nlr={self.lr[name]:g}"
                                   for name in reversed(self.names)], fontsize=9)
        seed_axes.set_xlabel("точність на тесті, %")
        seed_axes.set_title(f"кожен seed, епоха {epoch}")
        low = min(self.curves[n]["test_acc"][:, epoch - 1].min() for n in self.names) * 100
        seed_axes.set_xlim(low - 0.2, 98.6)
        seed_axes.set_ylim(-0.6, len(self.names) - 0.4)
        figure.tight_layout(pad=1.2, w_pad=2.0)
        return figure

    def readout(self, params=None):
        epoch, focus, metric = self.key(params or {})
        rows = [{"label": "ЕПОХА", "value": f"{epoch} з {self.epochs}"}]
        means = {}
        for name in self.names:
            accuracies = self.curves[name]["test_acc"][:, epoch - 1] * 100
            means[name] = accuracies.mean()
            rows.append({"label": f"{name.upper()}, ТЕСТ",
                         "value": f"{accuracies.mean():.2f} ± {accuracies.std(ddof=1):.2f} %"})
        best = max(means, key=means.get)
        rows.append({"label": "НАЙКРАЩИЙ", "value": best})
        return rows


class HistoryScene(_PlotScene):

    DATASETS = ("MNIST", "CIFAR-10")

    def __init__(self, options=None):
        super().__init__()
        mnist, cifar = _load("mnist.json"), _load("cifar.json")
        self.sets = [
            {"runs": mnist["runs"], "main": "improved",
             "title": "MNIST, ReLU 784-256-10, Adam"},
            {"runs": cifar["runs"], "main": cifar["best"],
             "title": f"CIFAR-10, {cifar['runs'][cifar['best']]['label']}"},
        ]
        for dataset in self.sets:
            history = dataset["runs"][dataset["main"]]["history"]
            dataset["epochs"] = len(history["val_loss"])
            dataset["best"] = int(np.argmin(history["val_loss"])) + 1

    def metadata(self):
        return {"description": "Криві навчання: train проти validation",
                "epochs": [dataset["epochs"] for dataset in self.sets]}

    def key(self, params):
        index = _int_param(params, "dataset", 0, 0, len(self.sets) - 1)
        limit = self.sets[index]["epochs"]
        return index, _int_param(params, "epoch", self.sets[index]["best"], 1, limit)

    def figure(self, index, epoch):
        dataset = self.sets[index]
        history = dataset["runs"][dataset["main"]]["history"]
        figure, (loss_axes, acc_axes) = imagekit.new_figure(*FIGURE_SIZE, ncols=2)
        epochs = np.arange(1, dataset["epochs"] + 1)

        loss_axes.plot(epochs, history["train_loss"], label="train", color=imagekit.ACCENT_HEX,
                       linewidth=2.0)
        loss_axes.plot(epochs, history["val_loss"], label="validation", color=imagekit.GOLD_HEX,
                       linewidth=2.0)
        best = dataset["best"]
        loss_axes.axvline(best, color=imagekit.GOLD_HEX, linewidth=1.0, linestyle=":")
        loss_axes.axvline(epoch, color=imagekit.MUTED_HEX, linewidth=1.0, linestyle="--")
        loss_axes.scatter([best], [history["val_loss"][best - 1]], color=imagekit.GOLD_HEX,
                          s=40, zorder=4, label=f"мін. validation, епоха {best}")
        loss_axes.set_yscale("log")
        loss_axes.set_xlabel("епоха")
        loss_axes.set_title(f"втрати: {dataset['title']}", fontsize=10)
        loss_axes.legend(loc="lower left", fontsize=9)

        for series, (key, run) in enumerate(dataset["runs"].items()):
            color = imagekit.SERIES_HEX[series % len(imagekit.SERIES_HEX)]
            accuracy = np.array(run["history"]["val_acc"]) * 100
            acc_axes.plot(np.arange(1, len(accuracy) + 1), accuracy, color=color,
                          linewidth=2.0 if key == dataset["main"] else 1.4, label=run["label"])
        acc_axes.axvline(epoch, color=imagekit.MUTED_HEX, linewidth=1.0, linestyle="--")
        acc_axes.set_xlabel("епоха")
        acc_axes.set_ylabel("%")
        acc_axes.set_title("точність на валідації", fontsize=10)
        if index == 0:
            acc_axes.set_ylim(90, 99)
        acc_axes.legend(loc="lower right", fontsize=8.5)
        figure.tight_layout(pad=1.2, w_pad=2.0)
        return figure

    def readout(self, params=None):
        index, epoch = self.key(params or {})
        dataset = self.sets[index]
        run = dataset["runs"][dataset["main"]]
        history = run["history"]
        at = epoch - 1
        best = dataset["best"]
        return [
            {"label": "НАБІР", "value": self.DATASETS[index]},
            {"label": "ЕПОХА", "value": f"{epoch} з {dataset['epochs']}"},
            {"label": "ВТРАТИ TRAIN / VAL",
             "value": f"{history['train_loss'][at]:.4f} / {history['val_loss'][at]:.4f}"},
            {"label": "ТОЧНІСТЬ TRAIN / VAL",
             "value": f"{history['train_acc'][at] * 100:.2f} / {history['val_acc'][at] * 100:.2f} %"},
            {"label": "ТОЧНІСТЬ НА ТЕСТІ", "value": f"{history['test_acc'][at] * 100:.2f} %"},
            {"label": "ОПТИМУМ (МІН. VAL LOSS)",
             "value": f"епоха {best}, тест {history['test_acc'][best - 1] * 100:.2f} %"},
        ]


def read_idx(path):
    raw = path.read_bytes()
    dims = raw[3]
    shape = tuple(int.from_bytes(raw[4 + 4 * i:8 + 4 * i], "big") for i in range(dims))
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * dims).reshape(shape)


class DigitNet:

    def __init__(self, path):
        weights = np.load(path)
        self.w1, self.b1 = weights["w1"], weights["b1"]
        self.w2, self.b2 = weights["w2"], weights["b2"]
        self.epoch = int(weights["epoch"])

    def probabilities(self, images):
        x = images.reshape(len(images), -1).astype(np.float32) / 255.0
        hidden = np.maximum(x @ self.w1.T + self.b1, 0.0)
        scores = hidden @ self.w2.T + self.b2
        scores -= scores.max(axis=1, keepdims=True)
        exp = np.exp(scores)
        return exp / exp.sum(axis=1, keepdims=True)


class DigitsScene(_PlotScene):
    STRIP = 12

    def __init__(self, options=None):
        super().__init__()
        self.images = read_idx(MNIST_RAW / "t10k-images-idx3-ubyte")
        self.labels = read_idx(MNIST_RAW / "t10k-labels-idx1-ubyte").astype(int)
        self.net = DigitNet(RESULTS / "mnist_mlp.npz")
        self.probs = self.net.probabilities(self.images)
        self.predicted = self.probs.argmax(axis=1)
        self.accuracy = float((self.predicted == self.labels).mean())

    def metadata(self):
        return {"description": "Передбачення мережі 784-256-10 на тестових цифрах",
                "count": len(self.labels), "accuracy": self.accuracy}

    def key(self, params):
        return (_int_param(params, "index", 0, 0, len(self.labels) - 1),)

    def figure(self, index):
        figure, _ = imagekit.new_figure(*FIGURE_SIZE)
        figure.clear()
        grid = figure.add_gridspec(2, self.STRIP, height_ratios=(3.2, 1.0),
                                   hspace=0.35, wspace=0.25)
        digit_axes = figure.add_subplot(grid[0, :5])
        bar_axes = figure.add_subplot(grid[0, 6:])

        truth, guess = self.labels[index], self.predicted[index]
        correct = truth == guess
        digit_axes.imshow(self.images[index], cmap="gray", interpolation="nearest")
        digit_axes.set_xticks([])
        digit_axes.set_yticks([])
        digit_axes.grid(False)
        for spine in digit_axes.spines.values():
            spine.set_color(imagekit.ACCENT_HEX if correct else ERROR_HEX)
            spine.set_linewidth(2.0)
        digit_axes.set_title(f"тест №{index}: справжня {truth}, мережа {guess}",
                             color=imagekit.TEXT_HEX if correct else ERROR_HEX)

        probs = self.probs[index] * 100
        colors = [imagekit.MUTED_HEX] * 10
        colors[truth] = imagekit.GOLD_HEX
        colors[guess] = imagekit.ACCENT_HEX if correct else ERROR_HEX
        bar_axes.set_axisbelow(True)
        bar_axes.grid(axis="x", visible=False)
        bar_axes.bar(range(10), probs, color=colors)
        bar_axes.set_xticks(range(10))
        bar_axes.set_ylim(0, 105)
        bar_axes.set_ylabel("%")
        bar_axes.set_title("softmax: ймовірність кожного класу")
        for digit in range(10):
            if probs[digit] >= 0.5:
                bar_axes.text(digit, probs[digit] + 1.5, f"{probs[digit]:.1f}",
                              ha="center", fontsize=8.5, color=imagekit.TEXT_HEX)

        for slot in range(self.STRIP):
            sample = (index + 1 + slot) % len(self.labels)
            axes = figure.add_subplot(grid[1, slot])
            axes.imshow(self.images[sample], cmap="gray", interpolation="nearest")
            axes.set_xticks([])
            axes.set_yticks([])
            axes.grid(False)
            ok = self.predicted[sample] == self.labels[sample]
            axes.set_title(f"№{sample}\nмережа: {self.predicted[sample]}",
                           fontsize=8.5, color=imagekit.TEXT_HEX if ok else ERROR_HEX)
            for spine in axes.spines.values():
                spine.set_color(imagekit.RULE_HEX if ok else ERROR_HEX)
        figure.subplots_adjust(left=0.04, right=0.98, top=0.93, bottom=0.03)
        return figure

    def readout(self, params=None):
        (index,) = self.key(params or {})
        truth, guess = self.labels[index], self.predicted[index]
        errors = int((self.predicted != self.labels).sum())
        return [
            {"label": "ТЕСТОВИЙ ПРИКЛАД", "value": f"№{index}"},
            {"label": "СПРАВЖНЯ ЦИФРА", "value": str(truth)},
            {"label": "ПЕРЕДБАЧЕННЯ", "value": f"{guess} ({self.probs[index, guess] * 100:.1f} %)"},
            {"label": "РЕЗУЛЬТАТ", "value": "вірно" if truth == guess else "помилка"},
            {"label": "ТОЧНІСТЬ НА ТЕСТІ",
             "value": f"{self.accuracy * 100:.2f} % ({errors} помилок з {len(self.labels)})"},
            {"label": "ВАГИ", "value": f"784-256-10, епоха {self.net.epoch}"},
        ]


SCENES = {"optimizers": OptimizersScene, "history": HistoryScene, "digits": DigitsScene}


def create_scene(name, options=None):
    if name not in SCENES:
        raise KeyError(f"unknown scene: {name}")
    return SCENES[name](options)
