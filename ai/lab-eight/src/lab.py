import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
OPTIMIZERS = ("SGD", "Momentum", "Nesterov", "Adagrad", "RMSprop", "Adam")


def _error_presets(count=4):
    try:
        errors = json.loads((RESULTS / "mnist.json").read_text())["errors"]
    except (OSError, KeyError, ValueError):
        return []
    step = max(len(errors) // count, 1)
    return [{"label": f"помилка №{index}", "params": {"index": index}}
            for index in errors[::step][:count]]


def manifest():
    return {
        "id": "ai-lab-eight",
        "course": "МТОІ",
        "courseTitle": "Методи та технології обчислювального інтелекту",
        "number": 8,
        "title": "Класифікація рукописних чисел повнозв'язною мережею. Методи оптимізації",
        "summary": "Варіант 18 (переліку варіантів у роботі немає — виконано повне завдання): "
                   "мережа з посібника покращена до 784-256-10 з ReLU і нормуванням, "
                   "порівняно шість методів градієнтного спуску на п'яти seed, torch.nn.Linear "
                   "написано з нуля. Той самий підхід перенесено на CIFAR-10.",
        "code": {"file": "fcnet.py", "label": "FCNET.PY", "library": "PYTORCH"},
        "demos": [
            {
                "id": "optimizers",
                "scene": "optimizers",
                "program": "optimizers_view.py",
                "title": "Методи оптимізації",
                "subtitle": "MNIST, 784-256-10 ReLU, 20 епох, по 5 seed на метод",
                "readout": {"label": "ТОЧНІСТЬ НА ТЕСТІ, СЕРЕДНЄ ± СКВ"},
                "controls": [
                    {"id": "epoch", "label": "ЕПОХА", "min": 1, "max": 20, "step": 1, "value": 20},
                    {"id": "focus", "label": "МЕТОД (0 — УСІ)", "min": 0,
                     "max": len(OPTIMIZERS), "step": 1, "value": 0},
                    {"id": "metric", "label": "КРИВА: ACC / LOSS / TRAIN", "min": 0, "max": 2,
                     "step": 1, "value": 0},
                ],
                "presets": [
                    {"label": "точність", "params": {"metric": 0, "focus": 0}},
                    {"label": "втрати val", "params": {"metric": 1, "focus": 0}},
                    {"label": "втрати train", "params": {"metric": 2, "focus": 0}},
                    {"label": "SGD", "params": {"focus": 1}},
                    {"label": "Adam", "params": {"focus": 6}},
                    {"label": "епоха 3", "params": {"epoch": 3}},
                ],
            },
            {
                "id": "history",
                "scene": "history",
                "program": "history_view.py",
                "title": "Train проти validation",
                "subtitle": "перенавчання й оптимальна кількість епох, MNIST і CIFAR-10",
                "readout": {"label": "КРИВІ НАВЧАННЯ"},
                "controls": [
                    {"id": "dataset", "label": "НАБІР: MNIST / CIFAR-10", "min": 0, "max": 1,
                     "step": 1, "value": 0},
                    {"id": "epoch", "label": "ЕПОХА", "min": 1, "max": 200, "step": 1,
                     "value": 40},
                ],
                "presets": [
                    {"label": "MNIST, 40 епох", "params": {"dataset": 0, "epoch": 40}},
                    {"label": "MNIST, 200 епох", "params": {"dataset": 0, "epoch": 200}},
                    {"label": "CIFAR-10, 40 епох", "params": {"dataset": 1, "epoch": 40}},
                ],
            },
            {
                "id": "digits",
                "scene": "digits",
                "program": "digits_view.py",
                "title": "Тестові цифри",
                "subtitle": "інференс збереженої мережі 784-256-10 прямо у сцені",
                "readout": {"label": "ПЕРЕДБАЧЕННЯ"},
                "controls": [
                    {"id": "index", "label": "ТЕСТОВИЙ ПРИКЛАД", "min": 0, "max": 9999,
                     "step": 1, "value": 0},
                ],
                "presets": [{"label": "№0", "params": {"index": 0}}] + _error_presets(),
            },
        ],
    }
