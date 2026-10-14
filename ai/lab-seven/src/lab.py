import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
WINE_EPOCHS = 5000
CANCER_EPOCHS = 300


def _experiments():
    try:
        return json.loads((RESULTS / "wine.json").read_text(encoding="utf-8"))["experiments"]
    except (OSError, ValueError, KeyError):
        return []


def _index(entries, key):
    return next((i for i, e in enumerate(entries) if e["key"] == key), 0)


def manifest():
    everything = _experiments()
    two_d = [e for e in everything if e["config"]["features"] == 2]
    epoch_control = {"id": "epoch", "label": "ЕПОХА", "min": 0, "max": WINE_EPOCHS,
                     "step": 50, "value": WINE_EPOCHS}

    return {
        "id": "ai-lab-seven",
        "course": "МТОІ",
        "courseTitle": "Методи та технології обчислювального інтелекту",
        "number": 7,
        "title": "Нейронна мережа для задачі класифікації",
        "summary": "Варіант 18: WineNet на датасеті вин (sklearn) і 28 його модифікацій "
                   "за пунктами a–g — ширина, глибина, оптимізатор, learning rate, "
                   "активація, усі 13 ознак, нормалізація. Друга частина — класифікація "
                   "пухлин Breast Cancer Wisconsin з відбором ознак за кореляцією Пірсона.",
        "code": {"file": "classifier.py", "label": "CLASSIFIER.PY", "library": "PYTORCH"},
        "demos": [
            {
                "id": "boundary",
                "scene": "boundary",
                "program": "boundary_view.py",
                "title": "Межі рішень на двох ознаках",
                "subtitle": "alcohol × malic_acid, заливка — клас і впевненість softmax",
                "readout": {"label": "МОДЕЛЬ І МЕТРИКИ"},
                "controls": [
                    {"id": "experiment", "label": "ЕКСПЕРИМЕНТ", "min": 0,
                     "max": max(len(two_d) - 1, 0), "step": 1, "value": 0},
                    epoch_control,
                ],
                "presets": [
                    {"label": "посібник", "params": {"experiment": _index(two_d, "base")}},
                    {"label": "до навчання", "params": {"experiment": _index(two_d, "base"),
                                                        "epoch": 0}},
                    {"label": "SGD", "params": {"experiment": _index(two_d, "c_sgd")}},
                    {"label": "lr 1e-2", "params": {"experiment": _index(two_d, "d_lr2")}},
                    {"label": "ReLU", "params": {"experiment": _index(two_d, "e_relu")}},
                    {"label": "перенавчання", "params": {"experiment": _index(two_d, "h_2d_tanh")}},
                ],
            },
            {
                "id": "sweep",
                "scene": "sweep",
                "program": "sweep_view.py",
                "title": "Порівняння модифікацій WineNet",
                "subtitle": "test accuracy кожної конфігурації, крива навчання і матриця помилок",
                "readout": {"label": "ЕКСПЕРИМЕНТ"},
                "controls": [
                    {"id": "experiment", "label": "ЕКСПЕРИМЕНТ", "min": 0,
                     "max": max(len(everything) - 1, 0), "step": 1, "value": 0},
                    epoch_control,
                ],
                "presets": [
                    {"label": "посібник", "params": {"experiment": _index(everything, "base")}},
                    {"label": "13 ознак", "params": {"experiment": _index(everything, "f_all")}},
                    {"label": "13 ознак + z-норм.",
                     "params": {"experiment": _index(everything, "g_all_std")}},
                    {"label": "4 шари", "params": {"experiment": _index(everything, "b_d4")}},
                    {"label": "1×10 ReLU", "params": {"experiment": _index(everything, "h_all_1x10")}},
                ],
            },
            {
                "id": "cancer",
                "scene": "cancer",
                "program": "cancer_view.py",
                "title": "Breast Cancer Wisconsin",
                "subtitle": "кореляції ознак, фільтр |r| > 0.9, втрати й матриця помилок",
                "readout": {"label": "КЛАСИФІКАЦІЯ ПУХЛИН"},
                "controls": [
                    {"id": "variant", "label": "ВАРІАНТ", "min": 0, "max": 2, "step": 1,
                     "value": 0},
                    {"id": "epoch", "label": "ЕПОХА", "min": 0, "max": CANCER_EPOCHS,
                     "step": 5, "value": CANCER_EPOCHS},
                ],
                "presets": [
                    {"label": "30 ознак", "params": {"variant": 0, "epoch": CANCER_EPOCHS}},
                    {"label": "після фільтра", "params": {"variant": 1, "epoch": CANCER_EPOCHS}},
                    {"label": "lr 1e-3", "params": {"variant": 2, "epoch": CANCER_EPOCHS}},
                    {"label": "lr 1e-3, епоха 30", "params": {"variant": 2, "epoch": 30}},
                ],
            },
        ],
    }
