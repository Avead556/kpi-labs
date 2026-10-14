import json
from pathlib import Path

import numpy as np

from labkit import imagekit
from labkit.imagekit import ImageQuad

RESULTS = Path(__file__).resolve().parents[1] / "results"
FIGURE_SIZE = (1440, 960)

NO_BN_HEX = imagekit.SERIES_HEX[2]
BN_HEX = imagekit.SERIES_HEX[0]
WRONG_HEX = "#D62728" if imagekit.LIGHT else "#D8735F"
RIGHT_HEX = "#2CA02C" if imagekit.LIGHT else "#8FB86B"

PAIRS = [
    ("mnist", "relu_3_max_pool", "relu_3_max_pool_bn", "MNIST · LeNet5 (relu, 2×3×3, max)"),
    ("cifar10", "relu_3_max_pool", "relu_3_max_pool_bn", "CIFAR-10 · LeNet5 (relu, 2×3×3, max)"),
    ("cifar10", "cifarnet_nobn", "cifarnet", "CIFAR-10 · CifarNet (3 блоки 16/32/64)"),
]

PARTS = [
    ("1 · Загасання градієнта: середній |∂y/∂w₁| для 17 активацій", None, None),
    ("2a · LeNet5 на MNIST: п'ять модифікацій з посібника", "mnist",
     ["tanh", "relu", "relu_3", "relu_3_max_pool", "relu_3_max_pool_bn"]),
    ("2b · Ті самі модифікації на CIFAR-10", "cifar10",
     ["tanh", "relu", "relu_3", "relu_3_max_pool", "relu_3_max_pool_bn"]),
    ("2c–2e · Більше фільтрів, padding, BN входу, CifarNet", "cifar10",
     ["relu_3_max_pool_bn", "wide", "wide_pad", "wide_pad_inbn", "cifarnet", "cifarnet_nobn"]),
    ("2g–2h · Dropout і L2 (weight_decay) для CifarNet", "cifar10",
     ["cifarnet", "cifarnet_dropout", "cifarnet_wd2", "cifarnet_wd3", "cifarnet_wd4",
      "cifarnet_wd5", "cifarnet_wd6"]),
]


class Results:

    _cache = {}

    @classmethod
    def get(cls, dataset):
        if dataset not in cls._cache:
            summary = json.loads((RESULTS / f"{dataset}.json").read_text(encoding="utf-8"))
            with np.load(RESULTS / f"{dataset}.npz") as arrays:
                summary["arrays"] = {key: arrays[key] for key in arrays.files}
            cls._cache[dataset] = summary
        return cls._cache[dataset]

    @classmethod
    def gradients(cls):
        if "gradients" not in cls._cache:
            cls._cache["gradients"] = json.loads(
                (RESULTS / "gradients.json").read_text(encoding="utf-8"))
        return cls._cache["gradients"]


def _missing_figure(size, title, missing):
    figure, axes = imagekit.new_figure(*size)
    axes.axis("off")
    axes.text(0.5, 0.5, "Немає збережених результатів для: " + ", ".join(missing),
              ha="center", va="center", fontsize=13, color=imagekit.MUTED_HEX,
              transform=axes.transAxes, wrap=True)
    figure.suptitle(title, fontsize=14, color=imagekit.TEXT_HEX)
    return figure


def _int(params, name, default, low, high):
    try:
        value = int(round(float((params or {}).get(name, default))))
    except (TypeError, ValueError):
        value = default
    return max(low, min(value, high))


def _epoch(params, data):
    return _int(params, "epoch", data["epochs"], 1, data["epochs"])


def _style_curve_axes(axes, data, epoch, ylabel):
    axes.set_xlabel("епоха")
    axes.set_ylabel(ylabel)
    axes.set_xlim(0.5, data["epochs"] + 0.5)
    axes.axvline(epoch, color=imagekit.MUTED_HEX, linewidth=1.2, linestyle=":")
    axes.xaxis.set_major_locator(_integer_locator())


def _integer_locator():
    from matplotlib.ticker import MaxNLocator
    return MaxNLocator(integer=True)


class FigureScene:

    def __init__(self, options=None):
        self.quad = ImageQuad()
        self._key = None

    def key(self, params):
        raise NotImplementedError

    def build(self, figure_size, params):
        raise NotImplementedError

    def draw(self, width, height, params=None):
        key = self.key(params or {})
        if key != self._key:
            with imagekit.style():
                figure = self.build(FIGURE_SIZE, params or {})
                self.quad.upload(imagekit.figure_to_rgb(figure))
            self._key = key
        self.quad.draw(width, height)


class BatchNormScene(FigureScene):

    def metadata(self):
        return {"description": "LeNet5 / CifarNet з BatchNorm і без"}

    def _pair(self, params):
        dataset, plain, normed, caption = PAIRS[_int(params, "pair", 0, 0, len(PAIRS) - 1)]
        data = Results.get(dataset)
        return data, data["runs"][plain], data["runs"][normed], caption

    def key(self, params):
        return (_int(params, "pair", 0, 0, 2), _int(params, "epoch", 20, 1, 99))

    def build(self, size, params):
        dataset, plain_id, normed_id, caption = PAIRS[_int(params, "pair", 0, 0, len(PAIRS) - 1)]
        missing = [r for r in (plain_id, normed_id) if r not in Results.get(dataset)["runs"]]
        if missing:
            return _missing_figure(size, caption, missing)
        data, plain, normed, caption = self._pair(params)
        epoch = _epoch(params, data)
        figure, (acc_axes, loss_axes) = imagekit.new_figure(*size, ncols=2)
        epochs = np.arange(1, data["epochs"] + 1)

        for run, color, name in ((plain, NO_BN_HEX, "без BN"), (normed, BN_HEX, "з BN")):
            acc = np.array(run["test_acc"]) * 100
            acc_axes.plot(epochs, acc, color=color, linewidth=2.2, marker="o",
                          markersize=4, label=name)
            acc_axes.plot(epoch, acc[epoch - 1], "o", color=color, markersize=11,
                          markeredgecolor=imagekit.TEXT_HEX)
            loss_axes.plot(epochs, run["test_loss"], color=color, linewidth=2.2,
                           label=f"{name}, тест")
            loss_axes.plot(epochs, run["train_loss"], color=color, linewidth=1.4,
                           linestyle="--", alpha=0.8, label=f"{name}, навчання")
            loss_axes.plot(epoch, run["test_loss"][epoch - 1], "o", color=color,
                           markersize=11, markeredgecolor=imagekit.TEXT_HEX)

        acc_axes.set_title("Точність на тесті, %")
        _style_curve_axes(acc_axes, data, epoch, "точність, %")
        acc_axes.legend(loc="lower right")
        loss_axes.set_title("Функція втрат (CrossEntropy)")
        _style_curve_axes(loss_axes, data, epoch, "втрати")
        loss_axes.legend(loc="upper right", fontsize=9)
        figure.suptitle(f"{caption} · епоха {epoch}", fontsize=14,
                        color=imagekit.TEXT_HEX)
        figure.tight_layout(rect=(0.01, 0.01, 0.99, 0.95), w_pad=3)
        return figure

    def readout(self, params=None):
        dataset, plain_id, normed_id, caption = PAIRS[_int(params, "pair", 0, 0, len(PAIRS) - 1)]
        if any(r not in Results.get(dataset)["runs"] for r in (plain_id, normed_id)):
            return [{"label": "НАБІР", "value": caption}, {"label": "СТАН", "value": "немає даних"}]
        data, plain, normed, caption = self._pair(params or {})
        epoch = _epoch(params, data)
        rows = [{"label": "НАБІР", "value": caption.split(" · ")[0]},
                {"label": "ЕПОХА", "value": f"{epoch} / {data['epochs']}"}]
        for run, name in ((plain, "БЕЗ BN"), (normed, "З BN")):
            best = int(np.argmax(run["test_acc"]))
            stop = int(np.argmin(run["test_loss"]))
            rows += [
                {"label": f"{name}: ТОЧНІСТЬ", "value": f"{run['test_acc'][epoch - 1] * 100:.2f} %"},
                {"label": f"{name}: ВТРАТИ", "value": f"{run['test_loss'][epoch - 1]:.4f}"},
                {"label": f"{name}: НАЙКРАЩА", "value": f"{run['test_acc'][best] * 100:.2f} % (еп. {best + 1})"},
                {"label": f"{name}: МІН. ВТРАТИ", "value": f"епоха {stop + 1}"},
            ]
        gain = (normed["test_acc"][epoch - 1] - plain["test_acc"][epoch - 1]) * 100
        rows.append({"label": "ВНЕСОК BN", "value": f"{gain:+.2f} в. п."})
        rows.append({"label": "ПАРАМЕТРІВ", "value": f"{plain['parameters']:,} / {normed['parameters']:,}".replace(",", " ")})
        return rows


PAGE_SIZE = 24
GRID = (4, 6)


class PredictionsScene(FigureScene):

    def metadata(self):
        return {"description": "Передбачення на тестових зображеннях"}

    def key(self, params):
        return (_int(params, "pair", 0, 0, 2), _int(params, "epoch", 20, 1, 99),
                _int(params, "page", 0, 0, 1))

    def _slice(self, params):
        dataset, plain, normed, caption = PAIRS[_int(params, "pair", 0, 0, len(PAIRS) - 1)]
        data = Results.get(dataset)
        epoch = _epoch(params, data)
        page = _int(params, "page", 0, 0, 1)
        rows = slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)
        arrays = data["arrays"]
        return (data, caption, epoch, page,
                arrays["probe_images"][rows], arrays["probe_labels"][rows],
                arrays[f"probe_{plain}"][epoch - 1][rows].astype(np.float32),
                arrays[f"probe_{normed}"][epoch - 1][rows].astype(np.float32))

    def build(self, size, params):
        dataset, plain_id, normed_id, caption = PAIRS[_int(params, "pair", 0, 0, len(PAIRS) - 1)]
        missing = [r for r in (plain_id, normed_id) if f"probe_{r}" not in Results.get(dataset)["arrays"]]
        if missing:
            return _missing_figure(size, caption, missing)
        data, caption, epoch, page, images, labels, p_plain, p_normed = self._slice(params)
        classes = data["classes"]
        stacked = max(len(name) for name in classes) > 6
        figure, grid = imagekit.new_figure(*size, nrows=GRID[0], ncols=GRID[1])
        for index, axes in enumerate(grid.flat):
            axes.set_xticks([])
            axes.set_yticks([])
            axes.grid(False)
            image = images[index]
            if image.shape[0] == 1:
                axes.imshow(image[0], cmap="gray", vmin=0, vmax=255)
            else:
                axes.imshow(image.transpose(1, 2, 0), interpolation="nearest")
            truth = int(labels[index])
            axes.set_title(f"істина: {classes[truth]}", fontsize=10,
                           color=imagekit.MUTED_HEX, pad=3)
            for column, (probs, x, align) in enumerate(
                    ((p_plain[index], 0.0, "left"), (p_normed[index], 1.0, "right"))):
                guess = int(np.argmax(probs))
                color = RIGHT_HEX if guess == truth else WRONG_HEX
                y = -0.04 - (0.13 * column if stacked else 0.0)
                axes.text(x, y, f"{classes[guess]} {probs[guess] * 100:.0f}%",
                          transform=axes.transAxes, ha=align, va="top", fontsize=9.5,
                          color=color)
        figure.suptitle(f"{caption} · епоха {epoch} · приклади {page * PAGE_SIZE + 1}–"
                        f"{(page + 1) * PAGE_SIZE}    ліворуч — без BN, праворуч — з BN",
                        fontsize=13, color=imagekit.TEXT_HEX)
        figure.subplots_adjust(left=0.02, right=0.98, top=0.9,
                               bottom=0.06 if stacked else 0.04,
                               wspace=0.25, hspace=0.6 if stacked else 0.45)
        return figure

    def readout(self, params=None):
        dataset, plain_id, normed_id, caption = PAIRS[_int(params, "pair", 0, 0, len(PAIRS) - 1)]
        if any(f"probe_{r}" not in Results.get(dataset)["arrays"] for r in (plain_id, normed_id)):
            return [{"label": "НАБІР", "value": caption}, {"label": "СТАН", "value": "немає даних"}]
        data, caption, epoch, page, images, labels, p_plain, p_normed = self._slice(params or {})
        dataset, plain, normed, _ = PAIRS[_int(params, "pair", 0, 0, len(PAIRS) - 1)]
        probe_labels = data["arrays"]["probe_labels"]
        rows = [{"label": "НАБІР · ЕПОХА", "value": f"{caption.split(' · ')[0]} · {epoch}"}]
        for run_id, name in ((plain, "БЕЗ BN"), (normed, "З BN")):
            probs = data["arrays"][f"probe_{run_id}"][epoch - 1].astype(np.float32)
            right_page = int((np.argmax(probs[page * PAGE_SIZE:(page + 1) * PAGE_SIZE], 1)
                              == labels).sum())
            right_all = int((np.argmax(probs, 1) == probe_labels).sum())
            rows += [
                {"label": f"{name}: НА СТОРІНЦІ", "value": f"{right_page} / {len(labels)}"},
                {"label": f"{name}: УСІ 48", "value": f"{right_all} / {len(probe_labels)}"},
                {"label": f"{name}: ТЕСТ (10 000)", "value": f"{data['runs'][run_id]['test_acc'][epoch - 1] * 100:.2f} %"},
            ]
        return rows


class ExperimentsScene(FigureScene):

    def metadata(self):
        return {"description": "Усі експерименти практичної частини"}

    def key(self, params):
        part = _int(params, "part", 1, 0, len(PARTS) - 1)
        return (part, 0 if part == 0 else _int(params, "epoch", 20, 1, 99))

    def build(self, size, params):
        part = _int(params, "part", 1, 0, len(PARTS) - 1)
        if part == 0:
            return self._gradients(size)
        title, dataset, run_ids = PARTS[part]
        data = Results.get(dataset)
        missing = [r for r in run_ids if r not in data["runs"]]
        run_ids = [r for r in run_ids if r in data["runs"]]
        if len(run_ids) < 2:
            return _missing_figure(size, title, missing)
        epoch = _epoch(params, data)
        figure, (acc_axes, loss_axes) = imagekit.new_figure(*size, ncols=2)
        epochs = np.arange(1, data["epochs"] + 1)
        colors = list(imagekit.SERIES_HEX) + ["#6B7780"]
        for index, run_id in enumerate(run_ids):
            run = data["runs"][run_id]
            color = colors[index % len(colors)]
            acc_axes.plot(epochs, np.array(run["test_acc"]) * 100, color=color,
                          linewidth=2, label=run["label"])
            loss_axes.plot(epochs, run["test_loss"], color=color, linewidth=2)
            stop = int(np.argmin(run["test_loss"]))
            loss_axes.plot(stop + 1, run["test_loss"][stop], "v", color=color, markersize=9)
        acc_axes.set_title("Точність на тесті, %")
        _style_curve_axes(acc_axes, data, epoch, "точність, %")
        acc_axes.legend(loc="lower right", fontsize=9)
        loss_axes.set_title("Втрати на тесті (▼ — мінімум, точка зупинки)")
        _style_curve_axes(loss_axes, data, epoch, "втрати")
        low = min(min(data["runs"][r]["test_loss"]) for r in run_ids)
        high = max(np.percentile(data["runs"][r]["test_loss"], 90) for r in run_ids)
        loss_axes.set_ylim(low * 0.9, high * 1.1)
        figure.suptitle(f"{title} · епоха {epoch}", fontsize=14, color=imagekit.TEXT_HEX)
        figure.tight_layout(rect=(0.01, 0.01, 0.99, 0.95), w_pad=3)
        return figure

    def _gradients(self, size):
        probe = Results.gradients()
        ranked = sorted(probe["mean_grad"].items(), key=lambda item: item[1])
        names = [name for name, _ in ranked]
        values = np.array([value for _, value in ranked])
        seeds = sorted(probe["per_seed"], key=int)
        spread = np.array([[probe["per_seed"][s][name] for s in seeds] for name in names])

        figure, axes = imagekit.new_figure(*size)
        colors = [imagekit.ACCENT_HEX if name == names[-1] else
                  imagekit.GOLD_HEX if name in ("Tanh", "Sigmoid") else imagekit.MUTED_HEX
                  for name in names]
        positions = np.arange(len(names))
        axes.barh(positions, values, color=colors, height=0.7)
        axes.errorbar(values, positions,
                      xerr=[values - spread.min(axis=1), spread.max(axis=1) - values],
                      fmt="none", ecolor=imagekit.TEXT_HEX, elinewidth=1, capsize=3)
        for y, value, right in zip(positions, values, spread.max(axis=1)):
            axes.text(max(value, right) * 1.12, y, f"{value:.4f}", va="center", fontsize=10)
        axes.set_yticks(positions, names)
        axes.set_xscale("log")
        axes.set_xlim(values.min() / 2, values.max() * 4)
        axes.set_xlabel(f"середнє |∂y/∂w₁| по {probe['experiments']} входах N(0, 1), "
                        f"seed {probe['seed']}; риски — розкид для seed {seeds[0]}–{seeds[-1]}")
        axes.grid(axis="y", visible=False)
        figure.suptitle(PARTS[0][0], fontsize=14, color=imagekit.TEXT_HEX)
        figure.tight_layout(rect=(0.01, 0.01, 0.99, 0.95))
        return figure

    def readout(self, params=None):
        part = _int(params, "part", 1, 0, len(PARTS) - 1)
        if part == 0:
            probe = Results.gradients()
            ranked = sorted(probe["mean_grad"].items(), key=lambda item: -item[1])
            rows = [{"label": name.upper(), "value": f"{value:.6f}"} for name, value in ranked[:6]]
            rows += [{"label": name.upper(), "value": f"{probe['mean_grad'][name]:.6f}"}
                     for name in ("Tanh", "Sigmoid")]
            return rows
        _, dataset, run_ids = PARTS[part]
        data = Results.get(dataset)
        run_ids = [r for r in run_ids if r in data["runs"]]
        epoch = _epoch(params, data)
        rows = [{"label": "ЕПОХА", "value": f"{epoch} / {data['epochs']}"}]
        for run_id in run_ids:
            run = data["runs"][run_id]
            stop = int(np.argmin(run["test_loss"]))
            rows.append({"label": run["label"].upper(),
                         "value": f"{run['test_acc'][epoch - 1] * 100:.2f} % · max "
                                  f"{max(run['test_acc']) * 100:.2f} · стоп еп. {stop + 1}"})
        return rows


SCENES = {
    "batchnorm": BatchNormScene,
    "predictions": PredictionsScene,
    "experiments": ExperimentsScene,
}


def create_scene(name, options=None):
    if name not in SCENES:
        raise KeyError(f"unknown scene: {name}")
    return SCENES[name](options)
