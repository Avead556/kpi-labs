import json
import re
from pathlib import Path

import numpy as np

from labkit import imagekit
from labkit.imagekit import ImageQuad

RESULTS = Path(__file__).resolve().parent.parent / "results"

DATASETS = ("mnist", "cifar")
DATASET_NAMES = {"mnist": "MNIST", "cifar": "CIFAR-10"}
MODEL_ORDER = ("classic", "sequential", "modular", "modular_lrelu", "modular_wide")
CIFAR_CLASSES = ("літак", "авто", "птах", "кіт", "олень",
                 "собака", "жаба", "кінь", "корабель", "вантажівка")

STYLE = {
    "classic": (imagekit.SERIES_HEX[0], "-", 2.4),
    "sequential": (imagekit.SERIES_HEX[1], (0, (5, 3)), 2.0),
    "modular": (imagekit.SERIES_HEX[2], (0, (1, 2.5)), 2.4),
    "modular_lrelu": (imagekit.SERIES_HEX[3], "-", 1.6),
    "modular_wide": (imagekit.SERIES_HEX[4], "-", 1.6),
}
FIGURE_SIZE = (1200, 760)
DIM_BAR_HEX = "#C8C8C8" if imagekit.LIGHT else imagekit.RULE_HEX
_DEFAULT_ARGUMENTS = re.compile(
    r", (eps=1e-05|momentum=0\.1|affine=True|bias=True|track_running_stats=True"
    r"|padding=0|dilation=1|ceil_mode=False|stride=\(1, 1\))(?=[,)])")


def _load_json(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def _dataset(params):
    index = int(round(float((params or {}).get("dataset", 0))))
    return DATASETS[max(0, min(index, len(DATASETS) - 1))]


def _clamped(params, key, low, high, default):
    value = int(round(float((params or {}).get(key, default))))
    return max(low, min(value, high))


def _parameter_groups(dataset, key, info):
    if "groups" in info:
        return info["groups"]
    import builders

    groups = {}
    for name, parameter in builders.build(dataset, key).named_parameters():
        parts = name.split(".")[:-1]
        depth = 3 if len(parts) >= 3 and parts[1] == "blocks" else 2
        group = ".".join(parts[:depth])
        groups[group] = groups.get(group, 0) + parameter.numel()
    return [[name, count] for name, count in groups.items()]


class _CachedPicture:

    def __init__(self):
        self.quad = ImageQuad()
        self.key = None

    def show(self, key, build, width, height):
        if key != self.key:
            self.quad.upload(build())
            self.key = key
        self.quad.draw(width, height)


class ArchitectureScene:

    def __init__(self, options=None):
        self.models = _load_json("models.json")
        self.history = _load_json("history.json")
        self.picture = _CachedPicture()

    def metadata(self):
        return {"description": "Архітектура моделей практичної №11"}

    def _pick(self, params):
        dataset = _dataset(params)
        keys = [key for key in MODEL_ORDER if key in self.models[dataset]]
        index = _clamped(params, "model", 0, len(keys) - 1, 0)
        return dataset, keys, keys[index]

    def draw(self, width, height, params=None):
        dataset, keys, key = self._pick(params)
        self.picture.show((dataset, key), lambda: self._figure(dataset, keys, key),
                          width, height)

    def _figure(self, dataset, keys, key):
        info = self.models[dataset][key]
        with imagekit.style():
            figure, _ = imagekit.new_figure(*FIGURE_SIZE, ncols=1)
            figure.clear()
            grid = figure.add_gridspec(2, 2, width_ratios=(1.15, 1), height_ratios=(1, 1.35),
                                       left=0.02, right=0.97, top=0.93, bottom=0.07,
                                       wspace=0.32, hspace=0.42)
            text_axes = figure.add_subplot(grid[:, 0])
            total_axes = figure.add_subplot(grid[0, 1])
            parts_axes = figure.add_subplot(grid[1, 1])

            figure.suptitle(f"{DATASET_NAMES[dataset]} · {info['label']} · "
                            f"{info['params']:,} параметрів".replace(",", " "),
                            x=0.02, ha="left", fontsize=13, color=imagekit.TEXT_HEX)
            self._draw_repr(text_axes, info["repr"])

            labels = [self.models[dataset][k]["label"] for k in keys]
            counts = [self.models[dataset][k]["params"] for k in keys]
            colors = [STYLE[k][0] if k == key else DIM_BAR_HEX for k in keys]
            positions = np.arange(len(keys))[::-1]
            total_axes.set_axisbelow(True)
            total_axes.barh(positions, counts, color=colors, height=0.62)
            total_axes.set_yticks(positions, labels, fontsize=9)
            total_axes.set_title("параметрів у моделі", fontsize=10, loc="left")
            total_axes.grid(axis="y", visible=False)
            total_axes.set_xlim(0, max(counts) * 1.32)
            for position, count, k in zip(positions, counts, keys):
                total_axes.text(count + max(counts) * 0.02, position, f"{count:,}".replace(",", " "),
                                va="center", fontsize=8.5,
                                color=imagekit.TEXT_HEX if k == key else imagekit.MUTED_HEX)
            total_axes.tick_params(axis="x", labelsize=8)

            groups = _parameter_groups(dataset, key, info)
            names = [name for name, _ in groups]
            values = [count for _, count in groups]
            positions = np.arange(len(groups))[::-1]
            parts_axes.set_axisbelow(True)
            parts_axes.barh(positions, values, color=STYLE[key][0], height=0.62)
            parts_axes.set_yticks(positions, names, fontsize=8.5)
            parts_axes.set_xscale("log")
            parts_axes.set_xlim(1, max(values) * 30)
            parts_axes.set_title("параметри за частинами (лог. шкала)", fontsize=10, loc="left")
            parts_axes.grid(axis="y", visible=False)
            parts_axes.tick_params(axis="x", labelsize=8)
            for position, value in zip(positions, values):
                parts_axes.text(value * 1.15, position, f"{value:,}".replace(",", " "),
                                va="center", fontsize=8, color=imagekit.TEXT_HEX)
            return imagekit.figure_to_rgb(figure)

    @staticmethod
    def _draw_repr(axes, text, max_lines=58, max_chars=64):
        lines = [_DEFAULT_ARGUMENTS.sub("", line) for line in text.splitlines()]
        lines = [line if len(line) <= max_chars else line[:max_chars - 1] + "…"
                 for line in lines]
        if len(lines) > max_lines:
            hidden = len(lines) - max_lines + 1
            lines = lines[:max_lines - 1] + [f"… ще {hidden} рядків"]
        axes.set_axis_off()
        axes.set_title("print(model), аргументи за замовчуванням прибрано", fontsize=10, loc="left", color=imagekit.MUTED_HEX)
        size = 8.6 if len(lines) <= 40 else max(6.2, 8.6 * 40 / len(lines))
        axes.text(0.0, 1.0, "\n".join(lines), family="DejaVu Sans Mono", fontsize=size,
                  va="top", ha="left", color=imagekit.TEXT_HEX, transform=axes.transAxes,
                  linespacing=1.18)

    def readout(self, params=None):
        dataset, _, key = self._pick(params)
        info = self.models[dataset][key]
        rows = [
            {"label": "СПОСІБ", "value": info["label"]},
            {"label": "КЛАС", "value": info["type"]},
            {"label": "ПАРАМЕТРІВ", "value": f"{info['params']:,}".replace(",", " ")},
            {"label": "МОДУЛІВ (УСЬОГО / ЛИСТКІВ)", "value": f"{info['modules']} / {info['leaves']}"},
            {"label": "КОНТЕЙНЕРИ", "value": ", ".join(info["containers"]) or "—"},
            {"label": "ШАРИ", "value": ", ".join(f"{name}×{count}"
                                                  for name, count in info["kinds"].items())},
        ]
        run = self.history.get(dataset, {}).get(key)
        if run:
            rows.append({"label": f"ТОЧНІСТЬ ПІСЛЯ {len(run['test_acc'])} ЕПОХ",
                         "value": f"{run['test_acc'][-1]:.2%} (макс. {max(run['test_acc']):.2%})"})
            rows.append({"label": "СЕРЕДНІЙ ЧАС ЕПОХИ", "value": f"{np.mean(run['seconds']):.1f} с"})
        return rows


class CurvesScene:

    def __init__(self, options=None):
        self.history = _load_json("history.json")
        self.picture = _CachedPicture()

    def metadata(self):
        return {"description": "Криві навчання практичної №11"}

    def _pick(self, params):
        dataset = _dataset(params)
        runs = self.history[dataset]
        epochs = max(len(run["test_acc"]) for run in runs.values())
        return dataset, runs, _clamped(params, "epoch", 1, epochs, epochs)

    def draw(self, width, height, params=None):
        dataset, runs, epoch = self._pick(params)
        self.picture.show((dataset, epoch), lambda: self._figure(dataset, runs, epoch),
                          width, height)

    def _figure(self, dataset, runs, epoch):
        with imagekit.style():
            figure, (acc_axes, loss_axes) = imagekit.new_figure(*FIGURE_SIZE, ncols=2)
            figure.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.1, wspace=0.22)
            figure.suptitle(f"{DATASET_NAMES[dataset]}: тест після кожної епохи "
                            f"(Adam, lr 1e-3, batch 100)",
                            x=0.07, ha="left", fontsize=13)
            for key in MODEL_ORDER:
                if key not in runs:
                    continue
                run = runs[key]
                color, dash, width = STYLE[key]
                epochs = np.arange(1, len(run["test_acc"]) + 1)
                acc_axes.plot(epochs, np.array(run["test_acc"]) * 100, color=color,
                              linestyle=dash, linewidth=width, label=run["label"])
                loss_axes.plot(epochs, run["test_loss"], color=color, linestyle=dash,
                               linewidth=width)
                if epoch <= len(epochs):
                    acc_axes.plot(epoch, run["test_acc"][epoch - 1] * 100, "o", color=color,
                                  markersize=7, markeredgecolor=imagekit.BACKGROUND_HEX,
                                  markeredgewidth=1.5)
                    loss_axes.plot(epoch, run["test_loss"][epoch - 1], "o", color=color,
                                   markersize=7, markeredgecolor=imagekit.BACKGROUND_HEX,
                                   markeredgewidth=1.5)
            for axes, title in ((acc_axes, "точність на тесті, %"),
                                (loss_axes, "втрати на тесті (cross-entropy)")):
                axes.axvline(epoch, color=imagekit.MUTED_HEX, linewidth=1, linestyle=":")
                axes.set_title(title, loc="left", fontsize=10.5)
                axes.set_xlabel("епоха")
                axes.xaxis.set_major_locator(_integer_locator())
            figure.legend(loc="upper left", bbox_to_anchor=(0.07, 0.935), ncol=5,
                          fontsize=9, frameon=False, handlelength=3)
            return imagekit.figure_to_rgb(figure)

    def readout(self, params=None):
        dataset, runs, epoch = self._pick(params)
        rows = [{"label": "ЕПОХА", "value": str(epoch)}]
        for key in MODEL_ORDER:
            if key in runs and epoch <= len(runs[key]["test_acc"]):
                run = runs[key]
                rows.append({"label": run["label"].upper(),
                             "value": f"{run['test_acc'][epoch - 1]:.2%} · "
                                      f"loss {run['test_loss'][epoch - 1]:.4f}"})
        same = [runs[key]["test_acc"][epoch - 1] for key in ("classic", "sequential", "modular")
                if key in runs and epoch <= len(runs[key]["test_acc"])]
        if len(same) == 3:
            rows.append({"label": "РОЗКИД ТРЬОХ ЗАПИСІВ",
                         "value": f"{(max(same) - min(same)) * 100:.2f} п.п."})
        return rows


def _integer_locator():
    from matplotlib.ticker import MaxNLocator
    return MaxNLocator(integer=True)


class TraceScene:

    CHANNELS_SHOWN = 8

    def __init__(self, options=None):
        import torch

        import builders

        torch.set_num_threads(1)
        self.torch = torch
        samples = np.load(RESULTS / "samples.npz")
        self.samples = {name: (samples[f"{name}_images"], samples[f"{name}_labels"])
                        for name in DATASETS}
        self.networks = {}
        for name in DATASETS:
            network = builders.build(name, "modular")
            state = torch.load(RESULTS / f"{name}_modular.pt", map_location="cpu")
            network.load_state_dict(state)
            self.networks[name] = network.eval()
        self.picture = _CachedPicture()
        self.cache = {}

    def metadata(self):
        return {"description": "Вихід кожного блоку енкодера (ModuleList)"}

    def _run(self, params):
        dataset = _dataset(params)
        images, labels = self.samples[dataset]
        index = _clamped(params, "sample", 0, len(images) - 1, 0)
        key = (dataset, index)
        if key not in self.cache:
            trace = []
            with self.torch.no_grad():
                logits = self.networks[dataset](self.torch.from_numpy(images[index:index + 1]),
                                                trace)
            probabilities = self.torch.softmax(logits, 1)[0].numpy()
            self.cache[key] = (images[index], int(labels[index]), probabilities,
                               [maps[0].numpy() for maps in trace])
        return dataset, index, self.cache[key]

    def draw(self, width, height, params=None):
        dataset, index, result = self._run(params)
        self.picture.show((dataset, index), lambda: self._figure(dataset, index, *result),
                          width, height)

    def _figure(self, dataset, index, image, label, probabilities, trace):
        names = _class_names(dataset)
        shown = self.CHANNELS_SHOWN
        with imagekit.style():
            figure, _ = imagekit.new_figure(*FIGURE_SIZE, ncols=1)
            figure.clear()
            outer = figure.add_gridspec(1, 3, width_ratios=(1.0, 2.6, 1.05), left=0.03,
                                        right=0.98, top=0.86, bottom=0.08, wspace=0.30)
            predicted = int(np.argmax(probabilities))
            figure.suptitle(f"{DATASET_NAMES[dataset]}, тестовий приклад #{index}: "
                            f"клас «{names[label]}», мережа каже «{names[predicted]}» "
                            f"({probabilities[predicted]:.1%})",
                            x=0.03, ha="left", fontsize=12.5)

            input_axes = figure.add_subplot(outer[0])
            picture = image[0] if image.shape[0] == 1 else np.transpose(image, (1, 2, 0))
            input_axes.imshow(picture, cmap="gray" if image.shape[0] == 1 else None,
                              interpolation="nearest")
            input_axes.set_title(f"вхід {tuple(image.shape)}", fontsize=9.5, loc="left")
            _bare(input_axes)

            rows = outer[1].subgridspec(len(trace), shown, hspace=0.55, wspace=0.08)
            for row, maps in enumerate(trace):
                for column in range(shown):
                    axes = figure.add_subplot(rows[row, column])
                    _bare(axes)
                    if column < maps.shape[0]:
                        axes.imshow(maps[column], cmap="magma", interpolation="nearest")
                    if column == 0:
                        axes.set_title(f"trace[{row}]: blocks[{row}] → {tuple(maps.shape)}, "
                                       f"канали 0–{min(shown, maps.shape[0]) - 1}",
                                       fontsize=9, loc="left", color=imagekit.MUTED_HEX)

            bars = figure.add_subplot(outer[2])
            positions = np.arange(len(names))[::-1]
            colors = [imagekit.ACCENT_HEX if i == predicted else DIM_BAR_HEX
                      for i in range(len(names))]
            bars.set_axisbelow(True)
            bars.barh(positions, probabilities, color=colors, height=0.62)
            bars.set_yticks(positions, [f"{name}{' ✓' if i == label else ''}"
                                        for i, name in enumerate(names)], fontsize=8.5)
            bars.set_xlim(0, 1)
            bars.grid(axis="y", visible=False)
            bars.set_title("softmax(decoder(…))", fontsize=9.5, loc="left")
            bars.tick_params(axis="x", labelsize=8)
            return imagekit.figure_to_rgb(figure)

    def readout(self, params=None):
        dataset, index, (image, label, probabilities, trace) = self._run(params)
        names = _class_names(dataset)
        predicted = int(np.argmax(probabilities))
        rows = [{"label": "СПРАВЖНІЙ КЛАС", "value": names[label]},
                {"label": "ПЕРЕДБАЧЕНО", "value": f"{names[predicted]} ({probabilities[predicted]:.2%})"},
                {"label": "ВХІД", "value": "×".join(map(str, image.shape))}]
        for row, maps in enumerate(trace):
            rows.append({"label": f"TRACE[{row}]", "value": "×".join(map(str, maps.shape))})
        flat = int(np.prod(trace[-1].shape))
        rows.append({"label": "FLATTEN → DECODER", "value": str(flat)})
        return rows


def _class_names(dataset):
    return CIFAR_CLASSES if dataset == "cifar" else tuple(str(digit) for digit in range(10))


def _bare(axes):
    axes.set_xticks([])
    axes.set_yticks([])
    axes.grid(False)
    for spine in axes.spines.values():
        spine.set_visible(False)


SCENES = {"arch": ArchitectureScene, "curves": CurvesScene, "trace": TraceScene}


def create_scene(name, options=None):
    if name not in SCENES:
        raise KeyError(f"unknown scene: {name}")
    return SCENES[name](options)
