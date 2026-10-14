import json
from pathlib import Path

import numpy as np

from labkit import imagekit
from labkit.imagekit import (ACCENT_HEX, GRID_HEX, MUTED_HEX, RULE_HEX, SERIES_HEX,
                             TEXT_HEX, ImageQuad)

RESULTS = Path(__file__).resolve().parents[1] / "results"
FIGURE_SIZE = (1200, 800)
CLASS_COLORS = SERIES_HEX[:3]
TRAIN_COLOR, TEST_COLOR = SERIES_HEX[0], SERIES_HEX[1]
GRID_STEPS = 220
CACHE_SIZE = 48

ACTIVATIONS = {
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
    "tanh": np.tanh,
    "relu": lambda x: np.maximum(x, 0.0),
    "elu": lambda x: np.where(x > 0, x, np.expm1(np.minimum(x, 0.0))),
    "leaky_relu": lambda x: np.where(x > 0, x, 0.01 * x),
}


def _nearest(values, target):
    return int(np.argmin(np.abs(np.asarray(values) - target)))


def _percent(value):
    return f"{100 * value:.1f} %"


class _CachedPicture:

    def __init__(self):
        self.quad = ImageQuad()
        self.cache = {}
        self.shown = None

    def show(self, key, render, width, height):
        if key != self.shown:
            if key not in self.cache:
                if len(self.cache) >= CACHE_SIZE:
                    self.cache.pop(next(iter(self.cache)))
                with imagekit.style():
                    self.cache[key] = render()
            self.quad.upload(self.cache[key])
            self.shown = key
        self.quad.draw(width, height)


def _epoch_curves(axes, epochs, series, marker, ylabel, legend_loc="best"):
    for (label, values), color in zip(series, (TRAIN_COLOR, TEST_COLOR)):
        axes.plot(epochs, values, color=color, linewidth=1.6, label=label)
    axes.axvline(marker, color=TEXT_HEX, linewidth=1.0, linestyle=(0, (4, 3)))
    axes.set_ylabel(ylabel)
    axes.legend(loc=legend_loc, fontsize=9)


def _confusion(axes, matrix, names, title):
    from matplotlib.colors import LinearSegmentedColormap

    ramp = LinearSegmentedColormap.from_list("counts", [GRID_HEX, ACCENT_HEX])
    axes.imshow(matrix, cmap=ramp, vmin=0, vmax=max(matrix.max(), 1))
    axes.grid(False)
    for (row, column), count in np.ndenumerate(matrix):
        dark = count > 0.55 * matrix.max()
        axes.text(column, row, str(count), ha="center", va="center", fontsize=13,
                  color=imagekit.BACKGROUND_HEX if dark else TEXT_HEX)
    axes.set_xticks(range(len(names)), names, fontsize=9)
    axes.set_yticks(range(len(names)), names, fontsize=9)
    axes.set_xlabel("передбачено")
    axes.set_ylabel("істина")
    axes.set_title(title, fontsize=10)


def _confusion_matrix(y_true, y_pred, n):
    matrix = np.zeros((n, n), dtype=int)
    np.add.at(matrix, (np.asarray(y_true), np.asarray(y_pred, dtype=int)), 1)
    return matrix


class _Wine:

    def __init__(self):
        self.meta = json.loads((RESULTS / "wine.json").read_text())
        self.experiments = self.meta["experiments"]
        curves = np.load(RESULTS / "wine_curves.npz")
        self.curves = {key: curves[key] for key in curves.files}
        self.class_names = ["клас 0", "клас 1", "клас 2"]

    def history(self, key):
        return {name: self.curves[f"{key}/{name}"]
                for name in ("epoch", "train_loss", "train_acc", "test_loss", "test_acc")}

    def at_epoch(self, key, epoch):
        history = self.history(key)
        i = _nearest(history["epoch"], epoch)
        snaps = self.curves[f"{key}/snap_epoch"]
        j = _nearest(snaps, epoch)
        return ({name: float(values[i]) for name, values in history.items()},
                int(snaps[j]), self.curves[f"{key}/test_pred"][j])

    def readout(self, entry, epoch):
        values, snap_epoch, _ = self.at_epoch(entry["key"], epoch)
        seeds = "  ".join(f"{a:.3f}" for a in entry["test_acc"])
        return [
            {"label": "ЕКСПЕРИМЕНТ", "value": f"{entry['item']}) {entry['label']}"},
            {"label": "КОНФІГУРАЦІЯ", "value": entry["description"]},
            {"label": "ЕПОХА", "value": f"{snap_epoch}"},
            {"label": "ACCURACY TRAIN / TEST", "value": f"{_percent(values['train_acc'])} / "
                                                        f"{_percent(values['test_acc'])}"},
            {"label": "LOSS TRAIN / TEST", "value": f"{values['train_loss']:.3f} / "
                                                    f"{values['test_loss']:.3f}"},
            {"label": "TEST ACC ПІСЛЯ 5000 ЕП. (3 SEED)", "value": seeds},
            {"label": "СЕРЕДНЄ ± СКВ", "value": f"{_percent(entry['test_mean'])} ± "
                                                f"{_percent(entry['test_std'])}"},
        ]


class BoundaryScene:

    def __init__(self, options=None):
        self.wine = _Wine()
        self.entries = [e for e in self.wine.experiments if e["config"]["features"] == 2]
        weights = np.load(RESULTS / "wine_weights.npz")
        self.weights = {key: weights[key] for key in weights.files}
        self.X_train = self.wine.curves["X_train"][:, :2]
        self.X_test = self.wine.curves["X_test"][:, :2]
        self.y_train, self.y_test = self.wine.curves["y_train"], self.wine.curves["y_test"]

        low = self.X_train.min(axis=0) - 1
        high = self.X_train.max(axis=0) + 1
        self.extent = (low[0], high[0], low[1], high[1])
        xs = np.linspace(low[0], high[0], GRID_STEPS)
        ys = np.linspace(low[1], high[1], GRID_STEPS)
        xx, yy = np.meshgrid(xs, ys)
        self.grid = np.column_stack([xx.ravel(), yy.ravel()])
        self.picture = _CachedPicture()

    def entry(self, params):
        index = int(round(float((params or {}).get("experiment", 0))))
        return self.entries[max(0, min(index, len(self.entries) - 1))]

    def regions(self, entry, snapshot):
        key = entry["key"]
        x = (self.grid - self.weights[f"{key}/mean"]) / self.weights[f"{key}/std"]
        activation = ACTIVATIONS[entry["config"]["activation"]]
        n_layers = len(entry["config"]["hidden"]) + 1
        for layer in range(n_layers):
            x = x @ self.weights[f"{key}/W{layer}"][snapshot].T + self.weights[f"{key}/b{layer}"][snapshot]
            if layer < n_layers - 1:
                x = activation(x)
        x = np.exp(x - x.max(axis=1, keepdims=True))
        probabilities = x / x.sum(axis=1, keepdims=True)
        shape = (GRID_STEPS, GRID_STEPS)
        return probabilities.argmax(1).reshape(shape), probabilities.max(1).reshape(shape)

    def render(self, entry, epoch):
        from matplotlib.colors import to_rgb

        key = entry["key"]
        snaps = self.wine.curves[f"{key}/snap_epoch"]
        snapshot = _nearest(snaps, epoch)
        classes, confidence = self.regions(entry, snapshot)

        figure, default_axes = imagekit.new_figure(*FIGURE_SIZE)
        default_axes.remove()
        grid = figure.add_gridspec(2, 2, width_ratios=(1.25, 1), hspace=0.32, wspace=0.22,
                                   left=0.07, right=0.98, top=0.9, bottom=0.08)
        plane = figure.add_subplot(grid[:, 0])
        loss_axes = figure.add_subplot(grid[0, 1])
        acc_axes = figure.add_subplot(grid[1, 1], sharex=loss_axes)

        palette = np.array([to_rgb(c) for c in CLASS_COLORS])
        background = np.array(to_rgb(imagekit.BACKGROUND_HEX))
        strength = 0.18 + 0.32 * (confidence[..., None] - 1 / 3) / (2 / 3)
        image = background + (palette[classes] - background) * strength
        plane.imshow(image, origin="lower", extent=self.extent, aspect="auto",
                     interpolation="nearest")
        plane.grid(False)
        for label, color in enumerate(CLASS_COLORS):
            train = self.y_train == label
            plane.scatter(*self.X_train[train].T, s=26, color=color,
                          edgecolors=imagekit.BACKGROUND_HEX, linewidths=0.8,
                          label=f"{self.wine.class_names[label]} (train)")
            test = self.y_test == label
            plane.scatter(*self.X_test[test].T, s=34, facecolors="none", edgecolors=color,
                          linewidths=1.4)
        plane.scatter([], [], s=34, facecolors="none", edgecolors=MUTED_HEX, label="test")
        plane.legend(loc="upper right", fontsize=9)
        plane.set_xlabel("alcohol")
        plane.set_ylabel("malic_acid")
        plane.set_title(f"межі рішень, епоха {int(snaps[snapshot])}", fontsize=11)

        history = self.wine.history(key)
        _epoch_curves(loss_axes, history["epoch"],
                      [("train", history["train_loss"]), ("test", history["test_loss"])],
                      int(snaps[snapshot]), "cross-entropy")
        _epoch_curves(acc_axes, history["epoch"],
                      [("train", history["train_acc"]), ("test", history["test_acc"])],
                      int(snaps[snapshot]), "accuracy", legend_loc="lower right")
        acc_axes.set_ylim(0.0, 1.02)
        acc_axes.set_xlabel("епоха")
        figure.suptitle(f"{entry['item']}) {entry['label']} — {entry['description']}",
                        fontsize=11, color=TEXT_HEX)
        return imagekit.figure_to_rgb(figure)

    def metadata(self):
        return {"description": "Межі рішень WineNet на двох ознаках",
                "experiments": [e["key"] for e in self.entries]}

    def draw(self, width, height, params=None):
        entry = self.entry(params)
        epoch = float((params or {}).get("epoch", 5000))
        snaps = self.wine.curves[f"{entry['key']}/snap_epoch"]
        snap = int(snaps[_nearest(snaps, epoch)])
        self.picture.show((entry["key"], snap), lambda: self.render(entry, snap), width, height)

    def readout(self, params=None):
        return self.wine.readout(self.entry(params), float((params or {}).get("epoch", 5000)))


class SweepScene:

    def __init__(self, options=None):
        self.wine = _Wine()
        self.entries = self.wine.experiments
        self.picture = _CachedPicture()

    def entry(self, params):
        index = int(round(float((params or {}).get("experiment", 0))))
        return self.entries[max(0, min(index, len(self.entries) - 1))]

    def render(self, entry, epoch):
        values, snap_epoch, test_pred = self.wine.at_epoch(entry["key"], epoch)
        figure, default_axes = imagekit.new_figure(*FIGURE_SIZE)
        default_axes.remove()
        grid = figure.add_gridspec(2, 2, width_ratios=(1.05, 1), hspace=0.38, wspace=0.3,
                                   left=0.29, right=0.97, top=0.93, bottom=0.08)
        bars = figure.add_subplot(grid[:, 0])
        curve = figure.add_subplot(grid[0, 1])
        matrix_axes = figure.add_subplot(grid[1, 1])

        rows = np.arange(len(self.entries))[::-1]
        means = np.array([e["test_mean"] for e in self.entries])
        spread = np.array([e["test_std"] for e in self.entries])
        colors = [ACCENT_HEX if e is entry else RULE_HEX for e in self.entries]
        bars.barh(rows, means, height=0.7, color=colors, xerr=spread,
                  error_kw={"ecolor": MUTED_HEX, "elinewidth": 1.0, "capsize": 2})
        bars.axvline(self.entries[0]["test_mean"], color=TEXT_HEX, linewidth=1.0,
                     linestyle=(0, (4, 3)))
        bars.set_yticks(rows, [f"{e['item']}) {e['label']}" for e in self.entries], fontsize=8)
        for row, mean in zip(rows, means):
            bars.text(min(mean, 1.0) + 0.012, row, f"{100 * mean:.1f}", va="center",
                      fontsize=7.5, color=MUTED_HEX)
        bars.set_xlim(0, 1.12)
        bars.set_ylim(-0.6, len(rows) - 0.4)
        bars.grid(axis="y", visible=False)
        bars.set_xlabel("test accuracy, середнє з 3 seed (пунктир — базова)")

        history = self.wine.history(entry["key"])
        _epoch_curves(curve, history["epoch"],
                      [("train", history["train_acc"]), ("test", history["test_acc"])],
                      snap_epoch, "accuracy", legend_loc="lower right")
        curve.set_ylim(0.0, 1.02)
        curve.set_xlabel("епоха")
        curve.set_title(f"{entry['item']}) {entry['label']}", fontsize=10)

        matrix = _confusion_matrix(self.wine.curves["y_test"], test_pred, 3)
        _confusion(matrix_axes, matrix, ["0", "1", "2"],
                   f"матриця помилок (test), епоха {snap_epoch}")
        return imagekit.figure_to_rgb(figure)

    def metadata(self):
        return {"description": "Порівняння модифікацій WineNet",
                "experiments": [e["key"] for e in self.entries]}

    def draw(self, width, height, params=None):
        entry = self.entry(params)
        epoch = float((params or {}).get("epoch", 5000))
        _, snap, _ = self.wine.at_epoch(entry["key"], epoch)
        self.picture.show((entry["key"], snap), lambda: self.render(entry, snap), width, height)

    def readout(self, params=None):
        entry = self.entry(params)
        epoch = float((params or {}).get("epoch", 5000))
        rows = self.wine.readout(entry, epoch)
        _, _, test_pred = self.wine.at_epoch(entry["key"], epoch)
        errors = int(np.sum(test_pred != self.wine.curves["y_test"]))
        rows.insert(4, {"label": "ПОМИЛОК НА TEST", "value": f"{errors} із {len(test_pred)}"})
        return rows


class CancerScene:

    VARIANTS = ("all", "filtered", "overfit")

    def __init__(self, options=None):
        self.meta = json.loads((RESULTS / "cancer.json").read_text())
        arrays = np.load(RESULTS / "cancer.npz")
        self.arrays = {key: arrays[key] for key in arrays.files}
        self.dropped = {name for name, _, _ in self.meta["dropped"]}
        self.picture = _CachedPicture()

    def variant(self, params):
        index = int(round(float((params or {}).get("variant", 0))))
        return self.VARIANTS[max(0, min(index, len(self.VARIANTS) - 1))]

    def at_epoch(self, variant, epoch):
        epochs = self.arrays[f"{variant}/epoch"]
        i = _nearest(epochs, epoch)
        snaps = self.arrays[f"{variant}/snap_epoch"]
        j = _nearest(snaps, epoch)
        values = {name: float(self.arrays[f"{variant}/{name}"][i])
                  for name in ("train_loss", "train_acc", "test_loss", "test_acc")}
        return values, int(snaps[j]), self.arrays[f"{variant}/test_pred"][j]

    def render(self, variant, epoch):
        from matplotlib.colors import LinearSegmentedColormap

        values, snap_epoch, test_pred = self.at_epoch(variant, epoch)
        summary = self.meta["variants"][variant]
        figure, default_axes = imagekit.new_figure(*FIGURE_SIZE)
        default_axes.remove()
        grid = figure.add_gridspec(2, 2, width_ratios=(1.3, 1), hspace=0.36, wspace=0.3,
                                   left=0.2, right=0.97, top=0.92, bottom=0.2)
        heat = figure.add_subplot(grid[:, 0])
        loss_axes = figure.add_subplot(grid[0, 1])
        matrix_axes = figure.add_subplot(grid[1, 1])

        ramp = LinearSegmentedColormap.from_list(
            "pearson", [SERIES_HEX[2], GRID_HEX, ACCENT_HEX])
        image = heat.imshow(self.arrays["corr"], cmap=ramp, vmin=-1, vmax=1)
        heat.grid(False)
        names = self.meta["feature_names"]
        labels = [("× " if n in self.dropped else "") + n for n in names]
        heat.set_xticks(range(len(names)), labels, rotation=90, fontsize=6.5)
        heat.set_yticks(range(len(names)), labels, fontsize=6.5)
        for tick_group in (heat.get_xticklabels(), heat.get_yticklabels()):
            for tick, name in zip(tick_group, names):
                tick.set_color(MUTED_HEX if name in self.dropped else TEXT_HEX)
        strong = np.argwhere(np.triu(np.abs(self.arrays["corr"]) > 0.9, k=1))
        heat.scatter(strong[:, 1], strong[:, 0], s=6, color="#0A0C0E", marker="s")
        heat.set_title(f"кореляція Пірсона (train); ■ — |r| > 0.9, × — видалено "
                       f"{len(self.dropped)}", fontsize=9.5)
        bar = figure.colorbar(image, ax=heat, fraction=0.04, pad=0.02)
        bar.outline.set_edgecolor(RULE_HEX)

        epochs = self.arrays[f"{variant}/epoch"]
        _epoch_curves(loss_axes, epochs,
                      [("train", self.arrays[f"{variant}/train_loss"]),
                       ("test", self.arrays[f"{variant}/test_loss"])],
                      snap_epoch, "cross-entropy", legend_loc="upper right")
        loss_axes.set_xlabel("епоха")
        loss_axes.set_title(f"{summary['label']}: {summary['n_features']} ознак", fontsize=10)

        matrix = _confusion_matrix(self.arrays["y_test"], test_pred, 2)
        _confusion(matrix_axes, matrix, ["M", "B"],
                   f"матриця помилок (test), епоха {snap_epoch}")
        return imagekit.figure_to_rgb(figure)

    def metadata(self):
        return {"description": "Breast Cancer Wisconsin: кореляції та класифікатор",
                "variants": list(self.VARIANTS)}

    def draw(self, width, height, params=None):
        variant = self.variant(params)
        epoch = float((params or {}).get("epoch", 300))
        _, snap, _ = self.at_epoch(variant, epoch)
        self.picture.show((variant, snap), lambda: self.render(variant, snap), width, height)

    def readout(self, params=None):
        variant = self.variant(params)
        values, snap_epoch, test_pred = self.at_epoch(variant, float((params or {}).get("epoch", 300)))
        summary = self.meta["variants"][variant]
        y_test = self.arrays["y_test"]
        missed = int(np.sum((y_test == 0) & (test_pred == 1)))
        false_alarm = int(np.sum((y_test == 1) & (test_pred == 0)))
        mean = summary["mean"]
        return [
            {"label": "ВАРІАНТ", "value": f"{summary['label']} ({summary['n_features']})"},
            {"label": "КОНФІГУРАЦІЯ", "value": summary["description"]},
            {"label": "ЕПОХА", "value": str(snap_epoch)},
            {"label": "ACCURACY TRAIN / TEST", "value": f"{_percent(values['train_acc'])} / "
                                                        f"{_percent(values['test_acc'])}"},
            {"label": "LOSS TRAIN / TEST", "value": f"{values['train_loss']:.3f} / "
                                                    f"{values['test_loss']:.3f}"},
            {"label": "ПРОПУЩЕНО M / ХИБНИХ M", "value": f"{missed} / {false_alarm}"},
            {"label": f"TEST ACC ПІСЛЯ {summary['epochs']} ЕП. (3 SEED)",
             "value": "  ".join(f"{a:.3f}" for a in summary["test_acc"])},
            {"label": "СЕРЕДНЄ: ACC / RECALL(M) / F1(M)",
             "value": f"{_percent(mean['accuracy'])} / {_percent(mean['recall'])} / "
                      f"{_percent(mean['f1'])}"},
        ]


SCENES = {"boundary": BoundaryScene, "sweep": SweepScene, "cancer": CancerScene}


def create_scene(name, options=None):
    if name not in SCENES:
        raise KeyError(f"unknown scene: {name}")
    return SCENES[name](options)
