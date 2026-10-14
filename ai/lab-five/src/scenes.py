import json
from pathlib import Path

import numpy as np

import gradient
import regression
from labkit import imagekit

RESULTS = Path(__file__).resolve().parents[1] / "results"

FUNCTION_ORDER = ("parabola", "skewed")
OPTIMIZER_ORDER = ("sgd", "momentum", "adam", "rmsprop")
VIEW = 10.0
GRID_TICKS = 240
PLOT_FLOOR = 1e-12
POINT_HEX = "#D8735F"
MAX_FIGURE = (1600, 1000)


def _pick(sequence, value):
    index = int(round(float(value)))
    return sequence[max(0, min(index, len(sequence) - 1))]


def _figure_size(width, height):
    width = max(640, min(int(width), MAX_FIGURE[0]))
    height = max(400, min(int(height), MAX_FIGURE[1]))
    return width, height


def _number(value, digits=3):
    return f"{value:.{digits}f}" if abs(value) < 1e4 else f"{value:.3g}"


def _float(params, name, default):
    return float((params or {}).get(name, default))


class DescentScene:

    def __init__(self, options=None):
        import torch

        torch.set_num_threads(1)
        self.quad = imagekit.ImageQuad()
        self._key = None
        self._run_key = None
        self._run = None
        axis = np.linspace(-VIEW, VIEW, GRID_TICKS)
        self.grid = np.meshgrid(axis, axis)
        self.surfaces = {name: gradient.FUNCTIONS[name][1](*self.grid)
                         for name in FUNCTION_ORDER}
        self._render((640, 400), *self.run({}))

    def metadata(self):
        return {"description": "Градієнтний спуск на двовимірній квадратичній функції"}

    def settings(self, params):
        return {
            "function": _pick(FUNCTION_ORDER, _float(params, "function", 0)),
            "optimizer": _pick(OPTIMIZER_ORDER, _float(params, "optimizer", 0)),
            "lr": 10.0 ** _float(params, "logLr", -3),
            "start": (_float(params, "x0", 8), _float(params, "y0", 8)),
            "steps": max(1, int(round(_float(params, "steps", 500)))),
        }

    def run(self, params):
        s = self.settings(params)
        key = (s["function"], s["optimizer"], s["lr"], s["start"], s["steps"])
        if key != self._run_key:
            path, values, diverged = gradient.descend(
                s["function"], list(s["start"]), s["lr"], s["steps"], s["optimizer"])
            self._run = (s, path, values, diverged)
            self._run_key = key
        return self._run

    def draw(self, width, height, params=None):
        size = _figure_size(width, height)
        run = self.run(params)
        key = (self._run_key, size)
        if key != self._key:
            self.quad.upload(self._render(size, *run))
            self._key = key
        self.quad.draw(width, height)

    def _render(self, size, s, path, values, diverged):
        with imagekit.style():
            figure, (contour, curve) = imagekit.new_figure(
                *size, ncols=2, gridspec_kw={"width_ratios": [1, 1]})
            figure.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.09, wspace=0.18)
            self._contours(contour, s, path, diverged)
            self._values(curve, s, values, diverged)
            rgb = imagekit.figure_to_rgb(figure)
            self.contour_box = contour.get_position().bounds
            return rgb

    def point_at(self, u, v, width, height):
        if self._key is None:
            return None
        figure_w, figure_h = self._key[1]
        image_aspect, view_aspect = figure_w / figure_h, width / max(height, 1)
        if image_aspect > view_aspect:
            v = 0.5 + (v - 0.5) * image_aspect / view_aspect
        else:
            u = 0.5 + (u - 0.5) * view_aspect / image_aspect
        left, bottom, box_w, box_h = self.contour_box
        px, py = (u - left) / box_w, (1 - v - bottom) / box_h
        if not (0 <= px <= 1 and 0 <= py <= 1):
            return None
        return (px * 2 - 1) * VIEW, (py * 2 - 1) * VIEW

    def _contours(self, axes, s, path, diverged):
        surface = self.surfaces[s["function"]]
        levels = np.geomspace(0.5, surface.max(), 22)
        axes.contour(*self.grid, surface, levels=levels, cmap="GnBu_r",
                     linewidths=0.8, alpha=0.75)
        outside = np.flatnonzero(np.abs(path).max(axis=1) > 3 * VIEW)
        shown = path[:outside[0] + 1] if outside.size else path
        axes.plot(shown[:, 0], shown[:, 1], color=imagekit.GOLD_HEX, linewidth=1.0, alpha=0.7)
        axes.scatter(shown[:, 0], shown[:, 1], s=9, color=POINT_HEX, zorder=3)
        axes.scatter(*path[0], s=60, color="white", zorder=4, label="старт")
        if not diverged:
            axes.scatter(*path[-1], s=60, color=imagekit.ACCENT_HEX, zorder=4,
                         edgecolors="white", label="кінець")
        axes.plot(0, 0, marker="+", markersize=14, color="white", linestyle="none",
                  label="мінімум (0, 0)")
        axes.set_xlim(-VIEW, VIEW)
        axes.set_ylim(-VIEW, VIEW)
        axes.set_aspect("equal")
        axes.set_xlabel("$x_1$")
        axes.set_ylabel("$x_2$")
        axes.set_title(f"f = {gradient.FUNCTIONS[s['function']][2]}", fontsize=10)
        axes.legend(loc="lower left", fontsize=8, framealpha=0.85)
        if diverged:
            axes.text(0, VIEW * 0.82, "РОЗБІЖНІСТЬ", ha="center", color=POINT_HEX,
                      fontsize=13, weight="bold",
                      bbox={"facecolor": imagekit.BACKGROUND_HEX, "edgecolor": POINT_HEX})

    def _values(self, axes, s, values, diverged):
        shown = np.where(values > 0, values, np.nan)
        axes.semilogy(np.arange(len(values)), shown, color=imagekit.ACCENT_HEX, linewidth=1.6)
        axes.set_ylim(bottom=max(np.nanmin(shown) if np.isfinite(shown).any() else 1.0,
                                 PLOT_FLOOR) / 3)
        axes.axhline(gradient.TOLERANCE, color=imagekit.MUTED_HEX, linestyle="--", linewidth=1)
        axes.text(len(values) - 1, gradient.TOLERANCE * 1.6, f"f = {gradient.TOLERANCE:g}",
                  ha="right", color=imagekit.MUTED_HEX, fontsize=8)
        reached = gradient.steps_to_tolerance(values)
        if reached is not None:
            axes.axvline(reached, color=imagekit.GOLD_HEX, linewidth=1, alpha=0.8)
            axes.text(reached, np.nanmax(shown), f" {reached}", color=imagekit.GOLD_HEX,
                      fontsize=9, va="top")
        axes.set_xlabel("крок")
        axes.set_ylabel("значення функції (log)")
        name = gradient.OPTIMIZERS[s["optimizer"]][0]
        axes.set_title(f"{name}, lr = {s['lr']:.3g}, кроків {s['steps']}", fontsize=10)
        axes.set_xlim(0, max(len(values) - 1, 1))

    def readout(self, params=None):
        s, path, values, diverged = self.run(params)
        reached = gradient.steps_to_tolerance(values)
        limit = gradient.sgd_stability_limit(s["function"])
        final = path[-1]
        rows = [
            {"label": "ФУНКЦІЯ", "value": gradient.FUNCTIONS[s["function"]][2]},
            {"label": "ОПТИМІЗАТОР", "value": gradient.OPTIMIZERS[s["optimizer"]][0]},
            {"label": "LEARNING RATE", "value": f"{s['lr']:.4g}"},
            {"label": "СТАРТ", "value": f"({s['start'][0]:.1f}, {s['start'][1]:.1f}),"
                                         f" f = {values[0]:.4g}"},
        ]
        if diverged:
            rows.append({"label": "РЕЗУЛЬТАТ",
                         "value": f"розбіжність на кроці {len(values) - 1}"})
        else:
            rows += [
                {"label": "ЗНАЙДЕНИЙ МІНІМУМ", "value": f"({final[0]:.4g}, {final[1]:.4g})"},
                {"label": "f У КІНЦІ", "value": f"{values[-1]:.3e}"},
            ]
        rows += [
            {"label": "ІТЕРАЦІЙ ВИКОНАНО", "value": str(len(values) - 1)},
            {"label": f"ІТЕРАЦІЙ ДО f < {gradient.TOLERANCE:g}",
             "value": str(reached) if reached is not None else "не досягнуто"},
            {"label": "ТОЧНИЙ МІНІМУМ", "value": "(0, 0), f = 0"},
            {"label": "МЕЖА lr ДЛЯ SGD", "value": f"2 / λmax = {limit:.4g}"},
        ]
        return rows


class RegressionScene:

    def __init__(self, options=None):
        self.quad = imagekit.ImageQuad()
        self._key = None
        data = np.load(RESULTS / "regression.npz")
        self.learning_rates = data["learning_rates"]
        self.losses = data["losses"]
        self.params = data["params"]
        self.summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
        self.optimal = [regression.optimal_epoch(losses) for losses in self.losses]
        self.last_finite = [int(np.flatnonzero(np.isfinite(losses))[-1]) for losses in self.losses]
        self._render((640, 400), *self.state({}))

    def metadata(self):
        return {"description": "Лінійна регресія врожайності, навчена градієнтним спуском"}

    def state(self, params):
        index = int(round(_float(params, "lr", 1)))
        index = max(0, min(index, len(self.learning_rates) - 1))
        epochs = self.losses.shape[1] - 1
        asked = max(0, min(int(round(_float(params, "epoch", 5000))), epochs))
        epoch = min(asked, self.last_finite[index])
        return index, asked, epoch

    def draw(self, width, height, params=None):
        size = _figure_size(width, height)
        key = (self.state(params), size)
        if key != self._key:
            self.quad.upload(self._render(size, *self.state(params)))
            self._key = key
        self.quad.draw(width, height)

    def _render(self, size, index, asked, epoch):
        losses = self.losses[index]
        params = self.params[index]
        steps = np.arange(len(losses))
        finite = np.isfinite(losses)

        with imagekit.style():
            figure, placeholder = imagekit.new_figure(*size)
            placeholder.remove()
            grid = figure.add_gridspec(2, 2, height_ratios=[1, 1],
                                       left=0.07, right=0.98, top=0.93, bottom=0.08,
                                       hspace=0.42, wspace=0.2)
            loss_axes = figure.add_subplot(grid[0, 0])
            weight_axes = figure.add_subplot(grid[0, 1])
            pred_axes = figure.add_subplot(grid[1, :])

            loss_axes.loglog(steps[finite] + 1, losses[finite], color=imagekit.ACCENT_HEX)
            loss_axes.axhline(self.summary["leastSquares"]["mse"], color=imagekit.MUTED_HEX,
                              linestyle="--", linewidth=1,
                              label=f"МНК: {self.summary['leastSquares']['mse']:.3f}")
            loss_axes.scatter([epoch + 1], [losses[epoch]], color="white", zorder=3, s=30)
            best = self.optimal[index]
            if best is not None:
                loss_axes.axvline(best + 1, color=imagekit.GOLD_HEX, linewidth=1,
                                  label=f"оптимум: {best} епох")
            loss_axes.legend(fontsize=8, loc="lower left", framealpha=0.85)
            loss_axes.set_xlabel("епоха + 1 (log)")
            loss_axes.set_ylabel("MSE (log)")
            loss_axes.set_title(f"lr = {self.learning_rates[index]:g}: MSE = {losses[epoch]:.4g}"
                                f" на епосі {epoch}", fontsize=10)

            labels = [f"w{row + 1}{col + 1}" for row in range(2) for col in range(3)] + ["b1", "b2"]
            span = finite if finite.all() else steps <= epoch
            for column, label in enumerate(labels):
                weight_axes.plot(steps[span], params[span, column], linewidth=1.2,
                                 label=label, linestyle="--" if label[0] == "b" else "-")
            weight_axes.axvline(epoch, color="white", linewidth=0.8, alpha=0.7)
            if finite.all():
                weight_axes.set_xscale("symlog", linthresh=10)
                weight_axes.set_xlim(0, len(losses) - 1)
            weight_axes.set_xlabel("епоха (symlog)" if finite.all() else "епоха")
            weight_axes.set_title("ваги та зміщення", fontsize=10)
            weight_axes.legend(ncol=4, fontsize=7, loc="upper left", framealpha=0.85)

            predictions = regression.predict(params[epoch])
            regions = np.arange(len(regression.TARGETS))
            bar = 0.2
            for k, (name, color) in enumerate(zip(("яблука", "апельсини"),
                                                  (imagekit.ACCENT_HEX, imagekit.GOLD_HEX))):
                offset = (k - 0.5) * 2 * bar
                pred_axes.bar(regions + offset - bar / 2, regression.TARGETS[:, k], bar,
                              color=color, alpha=0.35, label=f"{name}: ціль")
                pred_axes.bar(regions + offset + bar / 2, predictions[:, k], bar,
                              color=color, label=f"{name}: прогноз")
            low = min(0.0, float(np.nanmin(predictions)))
            high = max(140.0, float(np.nanmax(predictions)))
            pred_axes.set_ylim(low * 1.05, high * 1.3)
            pred_axes.set_xticks(regions, [f"регіон {r + 1}\n{tuple(int(v) for v in row)}"
                                           for r, row in enumerate(regression.INPUTS)])
            pred_axes.set_ylabel("врожайність")
            title = "прогноз і ціль (temp, rainfall, humidity)"
            if asked > epoch:
                title += f" — розбіжність після епохи {epoch}"
            pred_axes.set_title(title, fontsize=10)
            pred_axes.legend(ncol=4, fontsize=8, loc="upper left", framealpha=0.85)
            return imagekit.figure_to_rgb(figure)

    def readout(self, params=None):
        index, asked, epoch = self.state(params)
        losses = self.losses[index]
        values = self.params[index][epoch]
        best = self.optimal[index]
        lr = self.learning_rates[index]
        rows = [
            {"label": "LEARNING RATE", "value": f"{lr:g}"},
            {"label": "ЕПОХА", "value": str(epoch) + (" (далі розбіжність)" if asked > epoch else "")},
            {"label": "MSE", "value": _number(losses[epoch], 4)},
        ]
        if best is not None:
            rows.append({"label": "ОПТИМУМ ЕПОХ (≤ 1% від фіналу)",
                         "value": f"{best}, MSE = {losses[best]:.4f}"})
        elif self.last_finite[index] < len(losses) - 1:
            rows.append({"label": "ОПТИМУМ ЕПОХ", "value": "немає — lr вище межі стійкості"})
        else:
            rows.append({"label": "ОПТИМУМ ЕПОХ", "value": "немає плато за 20000 епох"})
        rows += [
            {"label": "MSE НАЙМЕНШИХ КВАДРАТІВ", "value": f"{self.summary['leastSquares']['mse']:.4f}"},
            {"label": "МЕЖА СТІЙКОСТІ lr", "value": f"{self.summary['stabilityLimit']:.4g}"},
            {"label": "w (яблука)", "value": ", ".join(_number(v) for v in values[0:3])},
            {"label": "w (апельсини)", "value": ", ".join(_number(v) for v in values[3:6])},
            {"label": "b", "value": ", ".join(_number(v) for v in values[6:8])},
        ]
        return rows


SCENES = {"descent": DescentScene, "regression": RegressionScene}


def create_scene(name, options=None):
    if name not in SCENES:
        raise KeyError(f"unknown scene: {name}")
    return SCENES[name](options)
