EPOCHS = 2000


def manifest():
    return {
        "id": "ai-lab-six",
        "course": "МТОІ",
        "courseTitle": "Методи та технології обчислювального інтелекту",
        "number": 6,
        "title": "Нейронна мережа для задачі регресії",
        "summary": "Варіант 18 (таблиці варіантів у роботі немає — виконано все завдання): "
                   "SineNet на PyTorch відновлює sin(x) із зашумлених точок, експерименти з "
                   "кількістю нейронів, епох, активацією, шарами та функцією втрат, а також "
                   "багатовимірна регресія на наборах «Енергоефективність» (UCI) і «Діабет».",
        "code": {"file": "regression.py", "label": "REGRESSION.PY", "library": "PYTORCH"},
        "demos": [
            {
                "id": "sine",
                "scene": "sine",
                "program": "sine_view.py",
                "title": "SineNet: навчання по епохах",
                "subtitle": "1 → 50 (tanh) → 1, Adam lr = 0.01, MSE, 100 зашумлених точок",
                "readout": {"label": "ПОХИБКИ НА ЕПОСІ"},
                "controls": [
                    {"id": "epoch", "label": "ЕПОХА", "min": 0, "max": EPOCHS, "step": 10,
                     "value": EPOCHS},
                ],
                "presets": [
                    {"label": "до навчання", "params": {"epoch": 0}},
                    {"label": "100", "params": {"epoch": 100}},
                    {"label": "500", "params": {"epoch": 500}},
                    {"label": "1000", "params": {"epoch": 1000}},
                    {"label": "2000", "params": {"epoch": 2000}},
                ],
                "stats": [
                    {"label": "ПАРАМЕТРІВ", "value": "151"},
                    {"label": "ШУМ", "value": "σ = 0.2"},
                    {"label": "SEED", "value": "18"},
                ],
            },
            {
                "id": "experiments",
                "scene": "experiments",
                "program": "experiments_view.py",
                "title": "Експерименти a)–e)",
                "subtitle": "змінюється одне: нейрони, епохи, активація, шари або функція втрат",
                "readout": {"label": "MSE / MAE НА ВАЛІДАЦІЇ"},
                "controls": [
                    {"id": "group", "label": "ЕКСПЕРИМЕНТ", "min": 1, "max": 5, "step": 1,
                     "value": 1},
                ],
                "presets": [
                    {"label": "a) нейрони", "params": {"group": 1}},
                    {"label": "b) епохи", "params": {"group": 2}},
                    {"label": "c) активація", "params": {"group": 3}},
                    {"label": "d) шари", "params": {"group": 4}},
                    {"label": "e) втрати", "params": {"group": 5}},
                ],
            },
            {
                "id": "tabular",
                "scene": "tabular",
                "program": "tabular_view.py",
                "title": "Багатовимірна регресія",
                "subtitle": "прогноз проти факту на тестовій вибірці (20 %) і криві втрат",
                "readout": {"label": "ТЕСТОВА ВИБІРКА"},
                "controls": [
                    {"id": "dataset", "label": "НАБІР: 1 ЕНЕРГІЯ, 2 ДІАБЕТ", "min": 1, "max": 2,
                     "step": 1, "value": 1},
                    {"id": "epoch", "label": "ЕПОХА", "min": 0, "max": EPOCHS, "step": 20,
                     "value": EPOCHS},
                ],
                "presets": [
                    {"label": "енергія", "params": {"dataset": 1, "epoch": 2000}},
                    {"label": "енергія, 100 епох", "params": {"dataset": 1, "epoch": 100}},
                    {"label": "діабет", "params": {"dataset": 2, "epoch": 500}},
                    {"label": "діабет, 2000 епох", "params": {"dataset": 2, "epoch": 2000}},
                ],
            },
        ],
    }
