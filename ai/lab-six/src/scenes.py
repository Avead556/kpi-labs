import json
from pathlib import Path

import numpy as np

from labkit import imagekit
from labkit.imagekit import (ACCENT_HEX, GOLD_HEX, MUTED_HEX, RULE_HEX, SERIES_HEX, TEXT_HEX,
                             ImageQuad)

RESULTS = Path(__file__).resolve().parents[1] / "results"
FIGURE_SIZE = (1280, 800)
DPI = 128
TRUTH_HEX = TEXT_HEX
ERROR_HEX = SERIES_HEX[2]

GROUPS = ["neurons", "epochs", "activation", "layers", "loss"]
DATASETS = ["energy", "diabetes"]
DATASET_TITLES = {"energy": "Енергоефективність (UCI)", "diabetes": "Діабет (sklearn)"}
TARGET_TITLES = {"Y1": "Y1 обігрів", "Y2": "Y2 охолодження", "progression": "прогресування"}


def _load_summary():
    return json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))


def _nearest(values, wanted):
    return int(np.abs(np.asarray(values) - wanted).argmin())


def _fmt(value, digits=4):
    return f"{value:.{digits}f}"


class _CachedFigureScene:

    def __init__(self):
        self.quad = ImageQuad()
        self._key = None

    def key(self, params):
        raise NotImplementedError

    def render(self, key):
        raise NotImplementedError

    def draw(self, width, height, params=None):
        key = self.key(params or {})
        if key != self._key:
            with imagekit.style():
                figure = self.render(key)
                self.quad.upload(imagekit.figure_to_rgb(figure))
            self._key = key
        self.quad.draw(width, height)


class SineScene(_CachedFigureScene):

    def __init__(self, options=None):
        super().__init__()
        data = np.load(RESULTS / "sine.npz")
        self.data = {name: data[name] for name in data.files}
        self.epochs = self.data["snapshot_epochs"]
        self.noise_mse = _load_summary()["sine"]["noise_mse"]

    def metadata(self):
        return {"epochs": int(self.epochs[-1]), "frames": len(self.epochs)}

    def key(self, params):
        return int(self.epochs[_nearest(self.epochs, float(params.get("epoch", self.epochs[-1])))])

    def _metrics(self, epoch):
        i = _nearest(self.epochs, epoch)
        d = self.data
        train_error = d["train_snapshots"][i] - d["y_train"]
        val_error = d["validation_snapshots"][i] - d["y_validation"]
        return {"train_mse": float((train_error ** 2).mean()),
                "val_mse": float((val_error ** 2).mean()),
                "val_mae": float(np.abs(val_error).mean())}

    def render(self, epoch):
        d = self.data
        i = _nearest(self.epochs, epoch)
        m = self._metrics(epoch)
        figure, (top, bottom) = imagekit.new_figure(
            *FIGURE_SIZE, dpi=DPI, nrows=2, gridspec_kw={"height_ratios": [3, 2]})

        top.scatter(d["x_train"], d["y_train"], s=14, color=GOLD_HEX, alpha=0.75,
                    linewidths=0, label="навчальні точки (sin + шум)")
        top.plot(d["grid"], np.sin(d["grid"]), color=TRUTH_HEX, lw=1.4, ls="--",
                 label="істинна функція sin(x)")
        top.plot(d["grid"], d["snapshots"][i], color=ACCENT_HEX, lw=2.2,
                 label="прогноз мережі")
        top.set_xlim(-10, 10)
        top.set_ylim(-1.9, 1.9)
        top.set_xlabel("x")
        top.set_ylabel("y")
        top.set_title(f"SineNet(50), tanh, Adam lr = 0.01 — епоха {epoch}", loc="left")
        top.legend(loc="lower left", fontsize=8, ncol=3, framealpha=0.9)

        epochs = np.arange(1, len(d["train_loss"]) + 1)
        bottom.plot(epochs, d["train_loss"], color=GOLD_HEX, lw=1.6, label="MSE на навчальних (з шумом)")
        bottom.plot(epochs, d["validation_mse"], color=ACCENT_HEX, lw=1.6, label="MSE на валідації (чистий sin)")
        bottom.axhline(self.noise_mse, color=MUTED_HEX, lw=1, ls=":",
                       label=f"дисперсія шуму {self.noise_mse:.3f}")
        bottom.set_yscale("log")
        bottom.set_xlim(0, epochs[-1])
        bottom.set_xlabel("епоха")
        bottom.set_ylabel("втрати (MSE)")
        bottom.axvline(epoch, color=TEXT_HEX, lw=1, alpha=0.6)
        if epoch > 0:
            bottom.plot([epoch], [d["train_loss"][epoch - 1]], "o", ms=8, color=GOLD_HEX,
                        mec=imagekit.BACKGROUND_HEX, mew=2)
            bottom.plot([epoch], [d["validation_mse"][epoch - 1]], "o", ms=8, color=ACCENT_HEX,
                        mec=imagekit.BACKGROUND_HEX, mew=2)
        bottom.legend(loc="upper right", fontsize=8, framealpha=0.9)
        bottom.text(0.01, 0.06, f"валідація: MSE {m['val_mse']:.4f}   MAE {m['val_mae']:.4f}",
                    transform=bottom.transAxes, color=TEXT_HEX, fontsize=9)
        figure.tight_layout()
        return figure

    def readout(self, params=None):
        epoch = self.key(params or {})
        m = self._metrics(epoch)
        return [
            {"label": "ЕПОХА", "value": f"{epoch} / {int(self.epochs[-1])}"},
            {"label": "MSE НАВЧ. (З ШУМОМ)", "value": _fmt(m["train_mse"])},
            {"label": "MSE ВАЛІДАЦІЯ", "value": _fmt(m["val_mse"])},
            {"label": "MAE ВАЛІДАЦІЯ", "value": _fmt(m["val_mae"])},
            {"label": "ДИСПЕРСІЯ ШУМУ", "value": _fmt(self.noise_mse)},
        ]


class ExperimentsScene(_CachedFigureScene):

    def __init__(self, options=None):
        super().__init__()
        data = np.load(RESULTS / "experiments.npz")
        self.arrays = {name: data[name] for name in data.files}
        self.table = _load_summary()["experiments"]
        sine = np.load(RESULTS / "sine.npz")
        self.grid, self.x_train, self.y_train = sine["grid"], sine["x_train"], sine["y_train"]

    def metadata(self):
        return {"groups": GROUPS}

    def key(self, params):
        index = int(round(float(params.get("group", 1)))) - 1
        return GROUPS[min(max(index, 0), len(GROUPS) - 1)]

    @staticmethod
    def _colors(group, n):
        if group in ("neurons", "epochs"):
            from matplotlib.colors import LinearSegmentedColormap
            stops = (["#9ECAE1", "#08306B"] if imagekit.LIGHT
                     else ["#23474A", ACCENT_HEX, "#E6FAFB"])
            ramp = LinearSegmentedColormap.from_list("sweep", stops)
            return [ramp(t) for t in np.linspace(0.0, 1.0, n)]
        return list(SERIES_HEX[:n])

    def render(self, group):
        rows = self.table[group]["rows"]
        predictions = self.arrays[f"{group}_predictions"]
        colors = self._colors(group, len(rows))
        figure, (left, right) = imagekit.new_figure(
            *FIGURE_SIZE, dpi=DPI, ncols=2, gridspec_kw={"width_ratios": [3, 2]})

        left.scatter(self.x_train, self.y_train, s=10, color=GOLD_HEX, alpha=0.35, linewidths=0)
        left.plot(self.grid, np.sin(self.grid), color=TRUTH_HEX, lw=1.4, ls="--", label="sin(x)")
        for row, prediction, color in zip(rows, predictions, colors):
            left.plot(self.grid, prediction, color=color, lw=1.6, label=row["label"])
        left.set_xlim(-10, 10)
        left.set_ylim(-1.9, 1.9)
        left.set_xlabel("x")
        left.set_ylabel("y")
        left.set_title(f"{self.table[group]['title']}: прогнози", loc="left")
        left.legend(loc="lower left", fontsize=7.5, ncol=min(6, len(rows) + 1), framealpha=0.9,
                    handlelength=1.2, columnspacing=0.8)

        values = [row["validation_mse"] for row in rows]
        best = int(np.argmin(values))
        positions = np.arange(len(rows))
        right.barh(positions, values, color=colors, height=0.7,
                   edgecolor=[TEXT_HEX if i == best else imagekit.BACKGROUND_HEX
                              for i in range(len(rows))], linewidth=1.5)
        right.set_yticks(positions, [row["label"] for row in rows])
        right.invert_yaxis()
        right.set_xlabel("MSE на валідації")
        right.set_title("помилка до чистого sin(x)", loc="left")
        for i, value in enumerate(values):
            right.text(value, i, f" {value:.3f}", va="center", fontsize=8,
                       color=TEXT_HEX if i == best else MUTED_HEX)
        right.set_xlim(0, max(values) * 1.3)
        right.grid(axis="y", visible=False)
        right.set_axisbelow(True)
        figure.tight_layout()
        return figure

    def readout(self, params=None):
        group = self.key(params or {})
        rows = self.table[group]["rows"]
        best = min(rows, key=lambda row: row["validation_mse"])
        out = [{"label": "ЕКСПЕРИМЕНТ", "value": self.table[group]["title"]},
               {"label": "НАЙКРАЩИЙ", "value": f"{best['label']} (MSE {best['validation_mse']:.4f})"}]
        out += [{"label": f"{row['label']}".upper(),
                 "value": f"MSE {row['validation_mse']:.4f} · MAE {row['validation_mae']:.4f}"}
                for row in rows]
        return out


class TabularScene(_CachedFigureScene):

    def __init__(self, options=None):
        super().__init__()
        data = np.load(RESULTS / "tabular.npz")
        self.arrays = {name: data[name] for name in data.files}
        self.summary = _load_summary()

    def metadata(self):
        return {"datasets": DATASETS}

    def key(self, params):
        index = int(round(float(params.get("dataset", 1)))) - 1
        name = DATASETS[min(max(index, 0), len(DATASETS) - 1)]
        epochs = self.arrays[f"{name}_snapshot_epochs"]
        epoch = int(epochs[_nearest(epochs, float(params.get("epoch", epochs[-1])))])
        return name, epoch

    def _metrics(self, name, epoch):
        epochs = self.arrays[f"{name}_snapshot_epochs"]
        prediction = self.arrays[f"{name}_snapshots"][_nearest(epochs, epoch)]
        target = self.arrays[f"{name}_test_target"]
        error = prediction - target
        total = ((target - target.mean(axis=0)) ** 2).sum(axis=0)
        return prediction, {"mse": (error ** 2).mean(axis=0), "mae": np.abs(error).mean(axis=0),
                            "r2": 1.0 - (error ** 2).sum(axis=0) / total}

    def render(self, key):
        name, epoch = key
        info = self.summary[name]
        prediction, metrics = self._metrics(name, epoch)
        target = self.arrays[f"{name}_test_target"]
        figure, (left, right) = imagekit.new_figure(*FIGURE_SIZE, dpi=DPI, ncols=2)

        low = float(min(target.min(), np.percentile(prediction, 1)))
        high = float(max(target.max(), np.percentile(prediction, 99)))
        pad = (high - low) * 0.05
        left.plot([low - pad, high + pad], [low - pad, high + pad], color=MUTED_HEX, lw=1, ls="--",
                  label="ідеал: прогноз = факт")
        for i, column in enumerate(info["targets"]):
            left.scatter(target[:, i], prediction[:, i], s=16, color=SERIES_HEX[i], alpha=0.85,
                         linewidths=0,
                         label=f"{TARGET_TITLES[column]}: MAE {metrics['mae'][i]:.2f}, "
                               f"R² {metrics['r2'][i]:.3f}")
        left.set_xlim(low - pad, high + pad)
        left.set_ylim(low - pad, high + pad)
        left.set_aspect("equal", adjustable="box")
        left.set_xlabel("фактичне значення (тест)")
        left.set_ylabel("прогноз мережі")
        left.set_title(f"{DATASET_TITLES[name]} — епоха {epoch}", loc="left")
        left.legend(loc="upper left", fontsize=8, framealpha=0.9)

        train_loss = self.arrays[f"{name}_train_loss"]
        val_mse = self.arrays[f"{name}_validation_mse"]
        best = int(self.arrays[f"{name}_best_epoch"])
        epochs = np.arange(1, len(train_loss) + 1)
        right.plot(epochs, train_loss, color=GOLD_HEX, lw=1.6, label="навчальна вибірка")
        right.plot(epochs, val_mse, color=ACCENT_HEX, lw=1.6, label="валідаційна вибірка")
        right.axvline(best, color=ERROR_HEX, lw=1, ls="--", label=f"мінімум валідації: {best}")
        right.axvline(epoch, color=TEXT_HEX, lw=1, alpha=0.6)
        if epoch > 0:
            for curve, color in ((train_loss, GOLD_HEX), (val_mse, ACCENT_HEX)):
                right.plot([epoch], [curve[epoch - 1]], "o", ms=8, color=color,
                           mec=imagekit.BACKGROUND_HEX, mew=2)
        right.set_yscale("log")
        right.set_xlim(0, epochs[-1])
        right.set_xlabel("епоха")
        right.set_ylabel("MSE (стандартизовані цілі)")
        hidden = info["chosen"]["hidden"]
        right.set_title(f"мережа {len(info['chosen']['features_list'])}→"
                        + "→".join(str(n) for n in hidden) + (f"→{len(info['targets'])}"),
                        loc="left")
        right.legend(loc="upper right", fontsize=8, framealpha=0.9)
        figure.tight_layout()
        return figure

    def readout(self, params=None):
        name, epoch = self.key(params or {})
        info = self.summary[name]
        _, metrics = self._metrics(name, epoch)
        rows = [{"label": "НАБІР", "value": DATASET_TITLES[name]},
                {"label": "ЕПОХА", "value": f"{epoch} / {info['epochs']}"},
                {"label": "ОЗНАКИ", "value": ", ".join(info["chosen"]["features_list"])},
                {"label": "НАЙКРАЩА ЕПОХА", "value": str(info["best_epoch"])}]
        for i, column in enumerate(info["targets"]):
            title = TARGET_TITLES[column].upper()
            rows += [{"label": f"{title}: MSE", "value": f"{metrics['mse'][i]:.3f}"},
                     {"label": f"{title}: MAE", "value": f"{metrics['mae'][i]:.3f}"},
                     {"label": f"{title}: R²", "value": f"{metrics['r2'][i]:.4f}"}]
        return rows


SCENES = {"sine": SineScene, "experiments": ExperimentsScene, "tabular": TabularScene}


def create_scene(name, options=None):
    if name not in SCENES:
        raise KeyError(f"unknown scene: {name}")
    return SCENES[name](options)
