MAX_EPOCH = 20

PAIRS = [
    {"label": "MNIST · LeNet5", "params": {"pair": 0}},
    {"label": "CIFAR-10 · LeNet5", "params": {"pair": 1}},
    {"label": "CIFAR-10 · CifarNet", "params": {"pair": 2}},
]


def _epoch(value=MAX_EPOCH):
    return {"id": "epoch", "label": "ЕПОХА", "min": 1, "max": MAX_EPOCH, "step": 1,
            "value": value}


def _pair():
    return {"id": "pair", "label": "ПАРА МОДЕЛЕЙ", "min": 0, "max": 2, "step": 1, "value": 0}


def manifest():
    return {
        "id": "ai-lab-ten",
        "course": "МТОІ",
        "courseTitle": "Методи та технології обчислювального інтелекту",
        "number": 10,
        "title": "Удосконалення згорткової нейронної мережі LeNet5. Batch Normalization",
        "summary": "Виконано практичну частину, крім необов'язкового CIFAR-100: пошук активації з найменшим "
                   "загасанням градієнта серед 17 функцій PyTorch, п'ять модифікацій "
                   "LeNet5 на MNIST і CIFAR-10, ширші фільтри, padding, BN входу, "
                   "запропонована мережа для CIFAR-10 з dropout і L2. Демонстрації "
                   "порівнюють навчання з пакетною нормалізацією і без неї.",
        "code": {"file": "lenet.py", "label": "LENET.PY", "library": "PYTORCH"},
        "demos": [
            {
                "id": "batchnorm",
                "scene": "batchnorm",
                "program": "batchnorm_view.py",
                "title": "LeNet5 з BatchNorm і без",
                "subtitle": "точність і втрати на тесті за епохами; однакова ініціалізація, seed 18",
                "readout": {"label": "НА ОБРАНІЙ ЕПОСІ"},
                "controls": [_pair(), _epoch()],
                "presets": PAIRS,
                "stats": [{"label": "ОПТИМІЗАТОР", "value": "Adam 1e-3"},
                          {"label": "БАТЧ", "value": "100"}],
            },
            {
                "id": "predictions",
                "scene": "predictions",
                "program": "predictions_view.py",
                "title": "Передбачення на тестових зображеннях",
                "subtitle": "24 фіксовані тестові приклади; ліворуч мітка без BN, праворуч — з BN",
                "readout": {"label": "ПЕРЕДБАЧЕННЯ"},
                "controls": [_pair(), _epoch(),
                             {"id": "page", "label": "СТОРІНКА ПРИКЛАДІВ", "min": 0, "max": 1,
                              "step": 1, "value": 0}],
                "presets": [{"label": "епоха 1", "params": {"epoch": 1}}, *PAIRS],
            },
            {
                "id": "experiments",
                "scene": "experiments",
                "program": "experiments_view.py",
                "title": "Усі експерименти практичної частини",
                "subtitle": "градієнти активацій; LeNet5 на MNIST і CIFAR-10; фільтри, padding, BN входу; dropout і L2",
                "readout": {"label": "ТОЧНІСТЬ НА ТЕСТІ"},
                "controls": [
                    {"id": "part", "label": "ПУНКТ", "min": 0, "max": 4, "step": 1, "value": 1},
                    _epoch(),
                ],
                "presets": [
                    {"label": "1 · градієнти", "params": {"part": 0}},
                    {"label": "2a · MNIST", "params": {"part": 1}},
                    {"label": "2b · CIFAR-10", "params": {"part": 2}},
                    {"label": "2c–e · архітектура", "params": {"part": 3}},
                    {"label": "2g–h · регуляризація", "params": {"part": 4}},
                ],
            },
        ],
    }
