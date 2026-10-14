import json
from pathlib import Path

import numpy as np

from labkit import imagekit

RESULTS = Path(__file__).resolve().parents[1] / "results"
FIGURE_SIZE = (1280, 800)
ERROR_HEX = imagekit.SERIES_HEX[2]
BAR_HEX = "#C7C7C7" if imagekit.LIGHT else imagekit.RULE_HEX

MODELS = [
    {"key": "lenet5", "title": "LeNet5 · MNIST", "weights": "lenet5_mnist.pt",
     "dataset": "mnist"},
    {"key": "lenet5plus", "title": "LeNet5Plus · MNIST", "weights": "lenet5plus_mnist.pt",
     "dataset": "mnist"},
    {"key": "cifar", "title": "LeNet5 · CIFAR-10", "weights": "lenet5_cifar.pt",
     "dataset": "cifar"},
]


def load_history():
    return json.loads((RESULTS / "history.json").read_text(encoding="utf-8"))


def _index(params, name, count, default=0):
    value = int(round(float((params or {}).get(name, default))))
    return max(0, min(value, count - 1))


def _feature_colormap():
    from matplotlib.colors import LinearSegmentedColormap

    stops = (["#F7FBFF", "#6BAED6", "#08306B"] if imagekit.LIGHT
             else [imagekit.BACKGROUND_HEX, "#164B52", imagekit.ACCENT_HEX, "#F2FBFB"])
    return LinearSegmentedColormap.from_list("hub", stops)


class CachedFigureScene:

    def __init__(self):
        self.quad = imagekit.ImageQuad()
        self._key = None

    def draw(self, width, height, params=None):
        key = self.picture_key(params or {})
        if key != self._key:
            self.quad.upload(self.render(*key))
            self._key = key
        self.quad.draw(width, height)


class PredictScene(CachedFigureScene):

    def __init__(self, options=None):
        super().__init__()
        import torch
        import lenet

        torch.set_num_threads(2)
        self.torch = torch
        self.history = load_history()
        samples = np.load(RESULTS / "samples.npz")
        self.images = {"mnist": samples["mnist_images"], "cifar": samples["cifar_images"]}
        self.labels = {"mnist": samples["mnist_labels"], "cifar": samples["cifar_labels"]}
        self.classes = {"mnist": lenet.MNIST_CLASSES, "cifar": lenet.CIFAR_CLASSES}

        factories = {"lenet5": lenet.LeNet5, "lenet5plus": lenet.LeNet5Plus,
                     "cifar": lambda: lenet.LeNet5(in_channels=3, padding=0)}
        self.models = []
        for spec in MODELS:
            model = factories[spec["key"]]()
            model.load_state_dict(torch.load(RESULTS / spec["weights"], weights_only=True))
            self.models.append(model.eval())
        self.errors = [self._errors(index) for index in range(len(MODELS))]
        self.colormap = _feature_colormap()

    def _batch(self, dataset, indexes):
        images = self.torch.from_numpy(self.images[dataset][indexes].astype(np.float32))
        return images.unsqueeze(1) if dataset == "mnist" else images

    def _errors(self, model_index):
        dataset = MODELS[model_index]["dataset"]
        with self.torch.no_grad():
            logits = self.models[model_index](self._batch(dataset, slice(None)))
        return int((logits.argmax(1).numpy() != self.labels[dataset]).sum())

    def metadata(self):
        return {"description": "Тестове зображення, ймовірності класів і карти ознак шару",
                "samples": int(len(self.labels["mnist"]))}

    def picture_key(self, params):
        model = _index(params, "model", len(MODELS))
        sample = _index(params, "sample", len(self.labels["mnist"]))
        layer = _index(params, "layer", 99)
        return model, sample, layer

    def _infer(self, model_index, sample, layer):
        dataset = MODELS[model_index]["dataset"]
        model = self.models[model_index]
        with self.torch.no_grad():
            x = self._batch(dataset, [sample])
            maps, _ = model.features(x)
            probabilities = self.torch.softmax(model(x), dim=1)[0].numpy()
        layer = min(layer, len(maps) - 1)
        name, activation = maps[layer]
        return dataset, probabilities, layer, len(maps), name, activation[0].numpy()

    def render(self, model_index, sample, layer):
        dataset, probabilities, layer, _, name, maps = self._infer(model_index, sample, layer)
        truth = int(self.labels[dataset][sample])
        predicted = int(np.argmax(probabilities))
        classes = self.classes[dataset]

        with imagekit.style():
            figure, _ = imagekit.new_figure(*FIGURE_SIZE)
            figure.clear()
            grid = figure.add_gridspec(2, 2, width_ratios=[1, 1.9], height_ratios=[1, 1],
                                       left=0.1, right=0.98, top=0.9, bottom=0.07,
                                       wspace=0.12, hspace=0.35)
            image_axes = figure.add_subplot(grid[0, 0])
            bar_axes = figure.add_subplot(grid[1, 0])
            map_axes = figure.add_subplot(grid[:, 1])

            image = self.images[dataset][sample]
            if dataset == "mnist":
                image_axes.imshow(image, cmap="gray", vmin=0, vmax=255)
            else:
                image_axes.imshow(np.transpose(image, (1, 2, 0)))
            image_axes.set_xticks([])
            image_axes.set_yticks([])
            image_axes.grid(False)
            verdict = "вірно" if truth == predicted else "ПОМИЛКА"
            image_axes.set_title(f"тест №{sample}: «{classes[truth]}» → «{classes[predicted]}»"
                                 f"  ({verdict})", fontsize=11,
                                 color=imagekit.TEXT_HEX if truth == predicted else ERROR_HEX)

            colors = [BAR_HEX] * 10
            colors[truth] = imagekit.GOLD_HEX
            colors[predicted] = imagekit.ACCENT_HEX if predicted == truth else ERROR_HEX
            positions = np.arange(10)
            bar_axes.barh(positions, probabilities, color=colors, height=0.7)
            bar_axes.set_yticks(positions, classes, fontsize=9)
            bar_axes.invert_yaxis()
            bar_axes.set_xlim(0, 1.18)
            bar_axes.set_xticks([0, 0.5, 1.0])
            bar_axes.grid(axis="y", visible=False)
            for position, value in zip(positions, probabilities):
                bar_axes.text(min(value, 1.0) + 0.02, position, f"{value:.3f}",
                              va="center", fontsize=8, color=imagekit.MUTED_HEX)
            bar_axes.set_title("softmax: ймовірності класів", fontsize=11)

            mosaic, rows, columns = _mosaic(maps)
            map_axes.imshow(mosaic, cmap=self.colormap, vmin=0, vmax=1,
                            interpolation="nearest")
            map_axes.set_xticks([])
            map_axes.set_yticks([])
            map_axes.grid(False)
            for spine in map_axes.spines.values():
                spine.set_visible(False)
            channels, height, width = maps.shape
            map_axes.set_title(f"шар {layer + 1}: {name} — {channels} карт {height}×{width}"
                               f"  (кожна нормована до [0, 1])", fontsize=11)
            figure.suptitle(MODELS[model_index]["title"], x=0.07, ha="left",
                            fontsize=13, color=imagekit.ACCENT_HEX)
            return imagekit.figure_to_rgb(figure)

    def readout(self, params=None):
        model_index, sample, layer = self.picture_key(params or {})
        dataset, probabilities, layer, layers, name, maps = self._infer(model_index, sample, layer)
        key = MODELS[model_index]["key"]
        history = self.history[key]
        classes = self.classes[dataset]
        truth = int(self.labels[dataset][sample])
        predicted = int(np.argmax(probabilities))
        best = int(np.argmax(history["test_acc"]))
        count = len(self.labels[dataset])
        return [
            {"label": "МОДЕЛЬ", "value": f"{MODELS[model_index]['title']}, "
                                          f"{history['params']:,} ваг".replace(",", " ")},
            {"label": "СПРАВЖНІЙ КЛАС", "value": classes[truth]},
            {"label": "ПЕРЕДБАЧЕННЯ", "value": f"{classes[predicted]} "
                                                f"(p = {probabilities[predicted]:.4f})"},
            {"label": "p СПРАВЖНЬОГО", "value": f"{probabilities[truth]:.4f}"},
            {"label": "ШАР", "value": f"{layer + 1}/{layers}: {name}, "
                                      f"{maps.shape[0]}×{maps.shape[1]}×{maps.shape[2]}"},
            {"label": "ТОЧНІСТЬ (ТЕСТ)", "value": f"{history['test_acc'][-1]:.4f} "
                                                   f"після {history['epochs']} еп."},
            {"label": "МАКСИМУМ", "value": f"{history['test_acc'][best]:.4f} "
                                            f"(епоха {best + 1})"},
            {"label": f"ПОМИЛОК СЕРЕД {count}", "value": str(self.errors[model_index])},
        ]


def _mosaic(maps, gap=2):
    channels, height, width = maps.shape
    columns = int(np.ceil(np.sqrt(channels * 1.6)))
    rows = int(np.ceil(channels / columns))
    mosaic = np.full((rows * (height + gap) - gap, columns * (width + gap) - gap), np.nan)
    for channel in range(channels):
        tile = maps[channel]
        low, high = tile.min(), tile.max()
        tile = (tile - low) / (high - low) if high > low else np.zeros_like(tile)
        row, column = divmod(channel, columns)
        top, left = row * (height + gap), column * (width + gap)
        mosaic[top:top + height, left:left + width] = tile
    return mosaic, rows, columns


class CurvesScene(CachedFigureScene):

    def __init__(self, options=None):
        super().__init__()
        self.history = load_history()

    def metadata(self):
        return {"description": "Криві навчання: втрати й точність за епохами"}

    def picture_key(self, params):
        model = _index(params, "model", len(MODELS))
        epochs = self.history[MODELS[model]["key"]]["epochs"]
        epoch = max(1, _index(params, "epoch", epochs + 1, default=epochs))
        return model, epoch

    def render(self, model_index, epoch):
        history = self.history[MODELS[model_index]["key"]]
        epochs = np.arange(1, history["epochs"] + 1)
        mnist = MODELS[model_index]["dataset"] == "mnist"

        with imagekit.style():
            figure, (loss_axes, acc_axes) = imagekit.new_figure(*FIGURE_SIZE, ncols=2)
            figure.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.1, wspace=0.22)

            loss_axes.plot(epochs, history["train_loss"], marker="o", markersize=3,
                           label="навчальна")
            loss_axes.plot(epochs, history["test_loss"], marker="o", markersize=3,
                           label="тестова")
            best_loss = int(np.argmin(history["test_loss"]))
            loss_axes.scatter([best_loss + 1], [history["test_loss"][best_loss]], s=90,
                              facecolors="none", edgecolors=imagekit.TEXT_HEX, zorder=5,
                              label=f"мін. тестова, еп. {best_loss + 1}")
            loss_axes.set_title("функція втрат (крос-ентропія)")
            loss_axes.set_xlabel("епоха")
            if mnist:
                loss_axes.set_yscale("log")

            acc_axes.plot(epochs, history["train_acc"], marker="o", markersize=3,
                          label="навчальна")
            acc_axes.plot(epochs, history["test_acc"], marker="o", markersize=3,
                          label="тестова")
            best_acc = int(np.argmax(history["test_acc"]))
            acc_axes.scatter([best_acc + 1], [history["test_acc"][best_acc]], s=90,
                             facecolors="none", edgecolors=imagekit.TEXT_HEX, zorder=5,
                             label=f"макс. тестова, еп. {best_acc + 1}")
            if mnist:
                acc_axes.axhline(0.992, color=ERROR_HEX, linestyle="--", linewidth=1,
                                 label="ціль 0.992")
                acc_axes.set_ylim(0.97, 1.0005)
            acc_axes.set_title("точність (accuracy)")
            acc_axes.set_xlabel("епоха")

            for axes in (loss_axes, acc_axes):
                axes.axvline(epoch, color=imagekit.MUTED_HEX, linewidth=1)
                axes.legend(fontsize=9, loc="best")
                axes.set_xticks(epochs if len(epochs) <= 12 else epochs[::2])
            figure.suptitle(f"{MODELS[model_index]['title']} — епоха {epoch}: тестова точність "
                            f"{history['test_acc'][epoch - 1]:.4f}", x=0.07, ha="left",
                            fontsize=13, color=imagekit.ACCENT_HEX)
            return imagekit.figure_to_rgb(figure)

    def readout(self, params=None):
        model_index, epoch = self.picture_key(params or {})
        history = self.history[MODELS[model_index]["key"]]
        at = epoch - 1
        best_acc = int(np.argmax(history["test_acc"]))
        best_loss = int(np.argmin(history["test_loss"]))
        gap = history["train_acc"][at] - history["test_acc"][at]
        return [
            {"label": "ЕПОХА", "value": f"{epoch} з {history['epochs']}"},
            {"label": "ВТРАТИ навч. / тест", "value": f"{history['train_loss'][at]:.4f} / "
                                                       f"{history['test_loss'][at]:.4f}"},
            {"label": "ТОЧНІСТЬ навч. / тест", "value": f"{history['train_acc'][at]:.4f} / "
                                                         f"{history['test_acc'][at]:.4f}"},
            {"label": "РОЗРИВ навч. − тест", "value": f"{gap * 100:+.2f} п.п."},
            {"label": "МАКС. ТОЧНІСТЬ", "value": f"{history['test_acc'][best_acc]:.4f} "
                                                  f"(епоха {best_acc + 1})"},
            {"label": "МІН. ТЕСТ. ВТРАТИ", "value": f"{history['test_loss'][best_loss]:.4f} "
                                                     f"(епоха {best_loss + 1})"},
            {"label": "ЧАС ЕПОХИ", "value": f"{np.median(history['seconds']):.1f} с "
                                             f"({self.history['device']})"},
        ]


DEVICES = ["mps", "cpu"]


class RunsScene(CachedFigureScene):

    def __init__(self, options=None):
        super().__init__()
        self.history = load_history()
        timed = {key.split("/")[1] for key in self.history["timing"]}
        self.devices = [device for device in DEVICES if device in timed]

    def metadata(self):
        return {"description": "Розкид точності між запусками і час шарів"}

    def picture_key(self, params):
        return (_index(params, "device", len(self.devices)),)

    def _timings(self, device_index):
        device = self.devices[device_index]
        return {key: self.history["timing"][f"{key}/{device}"]
                for key in ("lenet5", "lenet5plus")}

    def render(self, device_index):
        variance = self.history["variance"]
        timings = self._timings(device_index)

        with imagekit.style():
            figure, _ = imagekit.new_figure(*FIGURE_SIZE)
            figure.clear()
            grid = figure.add_gridspec(2, 2, width_ratios=[1, 1.35], left=0.09, right=0.98,
                                       top=0.93, bottom=0.08, wspace=0.55, hspace=0.35)
            spread_axes = figure.add_subplot(grid[:, 0])
            time_axes = [figure.add_subplot(grid[0, 1]), figure.add_subplot(grid[1, 1])]

            for position, key in enumerate(("lenet5", "lenet5plus")):
                runs = np.array(variance[key]["best"])
                jitter = np.linspace(-0.12, 0.12, len(runs))
                color = imagekit.SERIES_HEX[position]
                spread_axes.scatter(position + jitter, runs, color=color, s=40, zorder=3)
                mean, std = variance[key]["mean"], variance[key]["std"]
                spread_axes.errorbar(position + 0.3, mean, yerr=std, color=imagekit.TEXT_HEX,
                                     capsize=5, marker="_", markersize=14)
                spread_axes.text(position + 0.38, mean, f"{mean:.4f}\n±{std:.4f}",
                                 va="center", fontsize=9)
            spread_axes.axhline(0.992, color=ERROR_HEX, linestyle="--", linewidth=1)
            spread_axes.set_xticks([0, 1], [f"LeNet5\n{variance['lenet5']['epochs']} еп.",
                                            f"LeNet5Plus\n{variance['lenet5plus']['epochs']} еп."])
            spread_axes.set_xlim(-0.5, 1.9)
            spread_axes.set_title(f"макс. точність, {len(variance['lenet5']['best'])} seed'ів",
                                  fontsize=11)

            for axes, (key, rows) in zip(time_axes, timings.items()):
                rows = [row for row in rows if row["ms"] > 0]
                names = [row["layer"] for row in rows]
                values = [row["ms"] for row in rows]
                colors = [_layer_color(row["kind"]) for row in rows]
                positions = np.arange(len(rows))
                axes.barh(positions, values, color=colors, height=0.7)
                axes.set_yticks(positions, names, fontsize=7)
                axes.invert_yaxis()
                axes.grid(axis="y", visible=False)
                conv = sum(row["ms"] for row in rows if row["kind"] == "Conv2d")
                dense = sum(row["ms"] for row in rows if row["kind"] == "Linear")
                axes.set_title(f"{'LeNet5' if key == 'lenet5' else 'LeNet5Plus'}: "
                               f"згортки {conv:.2f} мс, повнозв'язні {dense:.2f} мс",
                               fontsize=10)
            time_axes[1].set_xlabel(f"мс на батч 100, прямий + зворотний прохід "
                                    f"({self.devices[device_index].upper()})", fontsize=9)
            return imagekit.figure_to_rgb(figure)

    def readout(self, params=None):
        (device_index,) = self.picture_key(params or {})
        variance = self.history["variance"]
        rows = []
        for key, title in (("lenet5", "LeNet5"), ("lenet5plus", "LeNet5Plus")):
            entry = variance[key]
            rows.append({"label": f"{title}: max acc", "value":
                         f"{entry['mean']:.4f} ± {entry['std']:.4f} "
                         f"(розмах {entry['spread']:.4f})"})
        for key, title in (("lenet5", "LeNet5"), ("lenet5plus", "LeNet5Plus")):
            timing = self._timings(device_index)[key]
            conv = sum(row["ms"] for row in timing if row["kind"] == "Conv2d")
            dense = sum(row["ms"] for row in timing if row["kind"] == "Linear")
            rows.append({"label": f"{title}: згортки / FC",
                         "value": f"{conv:.2f} / {dense:.2f} мс"})
        for delta, runs in variance["runs_needed"].items():
            rows.append({"label": f"ЗАПУСКІВ для Δ = {float(delta) * 100:g} п.п.",
                         "value": f"{runs} на архітектуру"})
        return rows


def _layer_color(kind):
    return {"Conv2d": imagekit.ACCENT_HEX, "Linear": imagekit.GOLD_HEX}.get(
        kind, imagekit.MUTED_HEX)


SCENES = {"predict": PredictScene, "curves": CurvesScene, "runs": RunsScene}


def create_scene(name, options=None):
    if name not in SCENES:
        raise KeyError(f"unknown scene: {name}")
    return SCENES[name](options)
