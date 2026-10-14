EPOCHS = 20000


def manifest():
    return {
        "id": "ai-lab-five",
        "course": "МТОІ",
        "courseTitle": "Методи та технології обчислювального інтелекту",
        "number": 5,
        "title": "Градієнтний спуск",
        "summary": "Варіант 18: лінійна регресія врожайності яблук і апельсинів за "
                   "температурою, опадами й вологістю, навчена градієнтним спуском на "
                   "PyTorch; оптимальна кількість епох — 5027 при lr = 1e-5 (MSE 0.537). "
                   "Поруч — живий градієнтний спуск на параболі й скошеній квадратичній "
                   "функції з вибором learning rate, старту й оптимізатора.",
        "code": {
            "file": "regression.py",
            "label": "REGRESSION.PY",
            "library": "PYTORCH",
        },
        "demos": [
            {
                "id": "descent",
                "scene": "descent",
                "program": "descent_view.py",
                "title": "Градієнтний спуск наживо",
                "subtitle": "траєкторія x на лініях рівня і спадання f(x) на кожному кроці",
                "readout": {"label": "РЕЗУЛЬТАТ СПУСКУ"},
                "controls": [
                    {"id": "function", "label": "ФУНКЦІЯ: 0 ПАРАБОЛА, 1 СКОШЕНА",
                     "min": 0, "max": 1, "step": 1, "value": 0},
                    {"id": "optimizer", "label": "0 SGD · 1 MOMENTUM · 2 ADAM · 3 RMSPROP",
                     "min": 0, "max": 3, "step": 1, "value": 0},
                    {"id": "logLr", "label": "LOG₁₀ LEARNING RATE",
                     "min": -4, "max": 0, "step": 0.05, "value": -3},
                    {"id": "x0", "label": "СТАРТ x₁", "min": -10, "max": 10,
                     "step": 0.5, "value": 8},
                    {"id": "y0", "label": "СТАРТ x₂", "min": -10, "max": 10,
                     "step": 0.5, "value": 8},
                    {"id": "steps", "label": "КРОКІВ", "min": 10, "max": 2000,
                     "step": 10, "value": 500},
                ],
                "presets": [
                    {"label": "посібник: парабола", "params": {
                        "function": 0, "optimizer": 0, "logLr": -3, "x0": 8, "y0": 8,
                        "steps": 500}},
                    {"label": "посібник: скошена", "params": {
                        "function": 1, "optimizer": 0, "logLr": -3, "x0": 8, "y0": 8,
                        "steps": 500}},
                    {"label": "скошена, lr 0.02", "params": {
                        "function": 1, "optimizer": 0, "logLr": -1.7, "x0": 8, "y0": 8,
                        "steps": 200}},
                    {"label": "розбіжність", "params": {
                        "function": 1, "optimizer": 0, "logLr": -1.6, "x0": 8, "y0": 8,
                        "steps": 200}},
                    {"label": "momentum", "params": {
                        "function": 1, "optimizer": 1, "logLr": -3, "x0": 8, "y0": 8,
                        "steps": 500}},
                    {"label": "adam", "params": {
                        "function": 1, "optimizer": 2, "logLr": -0.7, "x0": 8, "y0": 8,
                        "steps": 500}},
                ],
            },
            {
                "id": "regression",
                "scene": "regression",
                "program": "regression_view.py",
                "title": "Лінійна регресія: яблука й апельсини",
                "subtitle": f"MSE, ваги й прогнози на обраній епосі; {EPOCHS} епох навчання",
                "readout": {"label": "МОДЕЛЬ НА ЕПОСІ"},
                "controls": [
                    {"id": "lr", "label": "LR: 0 → 1e-6 · 1 → 1e-5 · 2 → 1e-4 · 3 → 1.2e-4",
                     "min": 0, "max": 3, "step": 1, "value": 1},
                    {"id": "epoch", "label": "ЕПОХА", "min": 0, "max": EPOCHS,
                     "step": 1, "value": 5027},
                ],
                "presets": [
                    {"label": "старт", "params": {"lr": 1, "epoch": 0}},
                    {"label": "100 епох", "params": {"lr": 1, "epoch": 100}},
                    {"label": "1000 епох", "params": {"lr": 1, "epoch": 1000}},
                    {"label": "оптимум 5027", "params": {"lr": 1, "epoch": 5027}},
                    {"label": "20000 епох", "params": {"lr": 1, "epoch": EPOCHS}},
                    {"label": "lr 1e-4", "params": {"lr": 2, "epoch": 518}},
                    {"label": "lr 1.2e-4", "params": {"lr": 3, "epoch": 15}},
                ],
            },
        ],
    }
