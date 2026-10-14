import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
SAMPLES = 500
MODEL_CONTROL = {"id": "model", "label": "МОДЕЛЬ (0 LeNet5 · 1 Plus · 2 CIFAR)",
                 "min": 0, "max": 2, "step": 1, "value": 0}


def _history():
    try:
        return json.loads((RESULTS / "history.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _best(history, key):
    runs = history.get(key)
    return f"{max(runs['test_acc']):.4f}" if runs else "—"


def manifest():
    history = _history()
    return {
        "id": "ai-lab-nine",
        "course": "МТОІ",
        "courseTitle": "Методи та технології обчислювального інтелекту",
        "number": 9,
        "title": "Розпізнавання рукописних чисел згортковою нейронною мережею",
        "summary": "LeNet5 на MNIST, покращена мережа з 3×3-згортками, BatchNorm і "
                   "max pooling для точності понад 0.992, розкид між запусками і час "
                   "шарів, LeNet5 на CIFAR-10 та підбір паддінгу. Навчання — заздалегідь, "
                   "сторінка показує передбачення, карти ознак і криві.",
        "code": {"file": "lenet.py", "label": "LENET.PY", "library": "PYTORCH"},
        "demos": [
            {
                "id": "predict",
                "scene": "predict",
                "program": "predict_view.py",
                "title": "Передбачення і карти ознак",
                "subtitle": f"перші {SAMPLES} зображень тестової вибірки, softmax і "
                            "активації обраного шару",
                "readout": {"label": "ПЕРЕДБАЧЕННЯ"},
                "controls": [
                    MODEL_CONTROL,
                    {"id": "sample", "label": "ТЕСТОВИЙ ПРИКЛАД №", "min": 0,
                     "max": SAMPLES - 1, "step": 1, "value": 0},
                    {"id": "layer", "label": "ШАР (карти ознак)", "min": 0, "max": 5,
                     "step": 1, "value": 0},
                ],
                "presets": [
                    {"label": "LeNet5 · conv1", "params": {"model": 0, "sample": 0, "layer": 0}},
                    {"label": "LeNet5 · conv2", "params": {"model": 0, "sample": 0, "layer": 2}},
                    {"label": "Plus · conv2b", "params": {"model": 1, "sample": 0, "layer": 4}},
                    {"label": "помилка LeNet5", "params": {"model": 0, "sample": 247, "layer": 0}},
                    {"label": "CIFAR-10", "params": {"model": 2, "sample": 3, "layer": 0}},
                ],
                "stats": [
                    {"label": "LeNet5, макс.", "value": _best(history, "lenet5")},
                    {"label": "LeNet5Plus, макс.", "value": _best(history, "lenet5plus")},
                    {"label": "CIFAR-10, макс.", "value": _best(history, "cifar")},
                ],
            },
            {
                "id": "curves",
                "scene": "curves",
                "program": "curves_view.py",
                "title": "Криві навчання",
                "subtitle": "втрати й точність на навчальній і тестовій вибірках за епохами",
                "readout": {"label": "ЕПОХА"},
                "controls": [
                    MODEL_CONTROL,
                    {"id": "epoch", "label": "ЕПОХА", "min": 1, "max": 25, "step": 1,
                     "value": 25},
                ],
                "presets": [
                    {"label": "LeNet5", "params": {"model": 0, "epoch": 20}},
                    {"label": "LeNet5Plus", "params": {"model": 1, "epoch": 10}},
                    {"label": "CIFAR-10", "params": {"model": 2, "epoch": 25}},
                ],
            },
            {
                "id": "runs",
                "scene": "runs",
                "program": "runs_view.py",
                "title": "Розкид між запусками і час шарів",
                "subtitle": "максимальна точність за 5 seed'ів; прямий і зворотний прохід "
                            "кожного шару на батчі 100",
                "readout": {"label": "ДИСПЕРСІЯ І ШВИДКІСТЬ"},
                "controls": [
                    {"id": "device", "label": "ПРИСТРІЙ (0 MPS · 1 CPU)", "min": 0, "max": 1,
                     "step": 1, "value": 0},
                ],
                "presets": [
                    {"label": "MPS", "params": {"device": 0}},
                    {"label": "CPU", "params": {"device": 1}},
                ],
            },
        ],
    }
