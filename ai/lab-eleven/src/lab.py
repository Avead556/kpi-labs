import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"

DATASET_CONTROL = {"id": "dataset", "label": "НАБІР: 0 MNIST, 1 CIFAR-10",
                   "min": 0, "max": 1, "step": 1, "value": 0}


def _stats():
    try:
        history = json.loads((RESULTS / "history.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    stats = []
    for dataset, name in (("mnist", "MNIST"), ("cifar", "CIFAR-10")):
        runs = history.get(dataset, {})
        if "classic" in runs and "modular" in runs:
            stats.append({"label": f"{name}: ПР №10 / МОДУЛІ",
                          "value": f"{runs['classic']['test_acc'][-1]:.2%} / "
                                   f"{runs['modular']['test_acc'][-1]:.2%}"})
    return stats


def manifest():
    return {
        "id": "ai-lab-eleven",
        "course": "МТОІ",
        "courseTitle": "Методи та технології обчислювального інтелекту",
        "number": 11,
        "title": "Способи побудови моделі у PyTorch",
        "summary": "Мережі практичної №10 — LeNet5 для MNIST і згорткова "
                   "мережа для CIFAR-10 — переписано трьома способами: nn.Module з "
                   "шаром на атрибут, один nn.Sequential і модульно (conv_block, "
                   "Encoder на ModuleList, Decoder, активації з ModuleDict). "
                   "Порівняно архітектуру, кількість параметрів і криві навчання.",
        "code": {"file": "builders.py", "label": "BUILDERS.PY", "library": "PYTORCH"},
        "stats": _stats(),
        "demos": [
            {
                "id": "arch",
                "scene": "arch",
                "program": "arch_view.py",
                "title": "Архітектура і параметри",
                "subtitle": "print(model) і параметри за частинами для кожного способу",
                "readout": {"label": "МОДЕЛЬ"},
                "controls": [
                    DATASET_CONTROL,
                    {"id": "model", "label": "СПОСІБ: 0 Module, 1 Sequential, 2–4 модулі",
                     "min": 0, "max": 4, "step": 1, "value": 0},
                ],
                "presets": [
                    {"label": "MNIST · nn.Module", "params": {"dataset": 0, "model": 0}},
                    {"label": "MNIST · Sequential", "params": {"dataset": 0, "model": 1}},
                    {"label": "MNIST · модулі", "params": {"dataset": 0, "model": 2}},
                    {"label": "CIFAR · nn.Module", "params": {"dataset": 1, "model": 0}},
                    {"label": "CIFAR · модулі", "params": {"dataset": 1, "model": 2}},
                    {"label": "CIFAR · 32-64-128", "params": {"dataset": 1, "model": 4}},
                ],
            },
            {
                "id": "curves",
                "scene": "curves",
                "program": "curves_view.py",
                "title": "Криві навчання",
                "subtitle": "точність і втрати на тесті за епохами, усі способи разом",
                "readout": {"label": "ТОЧНІСТЬ НА ЕПОСІ"},
                "controls": [
                    DATASET_CONTROL,
                    {"id": "epoch", "label": "ЕПОХА", "min": 1, "max": 10, "step": 1,
                     "value": 10},
                ],
                "presets": [
                    {"label": "MNIST", "params": {"dataset": 0, "epoch": 10}},
                    {"label": "MNIST · 1 епоха", "params": {"dataset": 0, "epoch": 1}},
                    {"label": "CIFAR-10", "params": {"dataset": 1, "epoch": 8}},
                    {"label": "CIFAR · 3 епохи", "params": {"dataset": 1, "epoch": 3}},
                ],
            },
            {
                "id": "trace",
                "scene": "trace",
                "program": "trace_view.py",
                "title": "ModuleList: вихід кожного блоку",
                "subtitle": "навчена модульна мережа, trace з енкодера для тестового зображення",
                "readout": {"label": "ПЕРЕДБАЧЕННЯ"},
                "controls": [
                    DATASET_CONTROL,
                    {"id": "sample", "label": "ТЕСТОВИЙ ПРИКЛАД", "min": 0, "max": 31,
                     "step": 1, "value": 0},
                ],
                "presets": [
                    {"label": "MNIST #0", "params": {"dataset": 0, "sample": 0}},
                    {"label": "MNIST #8", "params": {"dataset": 0, "sample": 8}},
                    {"label": "CIFAR #3", "params": {"dataset": 1, "sample": 3}},
                    {"label": "CIFAR #12", "params": {"dataset": 1, "sample": 12}},
                ],
            },
        ],
    }
