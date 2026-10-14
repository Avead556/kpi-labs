import dicom_data


def manifest():
    values, dataset = dicom_data.load_dicom()
    info = dicom_data.metadata(dataset, values)
    low, high = info["minValue"], info["maxValue"]
    span = max(high - low, 1.0)

    return {
        "id": "mv-lab-one",
        "course": "MV",
        "courseTitle": "Медична візуалізація",
        "number": 1,
        "title": "Сучасні бібліотеки для роботи з графічними даними",
        "summary": "Вибір мови, середовища та бібліотек для роботи з графічними "
                   "даними і файлами DICOM. Працездатність стека підтверджено двома "
                   "програмами: виведенням графічного примітиву та відображенням "
                   "медичного знімка.",
        "conclusion": "Обидві перевірки проходять на одному стеку: PyOpenGL із GLFW "
                      "дають контекст OpenGL 3.3 Core Profile, pydicom читає набір "
                      "даних і переводить збережені значення в одиниці Гаунсфілда. "
                      "Оконне перетворення винесене у фрагментний шейдер, тому зміна "
                      "вікна не потребує перерахунку зображення на боці CPU.",
        "code": {
            "file": "dicom_data.py",
            "label": "DICOM_DATA.PY",
            "library": "PYDICOM",
        },
        "demos": [
            {
                "id": "primitive",
                "scene": "primitive",
                "program": "primitive.py",
                "title": "Графічний примітив",
                "subtitle": "OpenGL 3.3 Core Profile: трикутник з інтерполяцією кольору",
                "animate": {"param": "spin", "speed": 0.6},
                "controls": [
                    {
                        "id": "spin",
                        "label": "ПОВОРОТ",
                        "unit": " рад",
                        "min": 0.0,
                        "max": 6.28,
                        "step": 0.01,
                        "value": 0.0,
                    },
                ],
                "note": "Колір задано у вершинах, проміжні значення отримані "
                        "інтерполяцією під час растеризації.",
            },
            {
                "id": "dicom",
                "scene": "dicom",
                "program": "dicom_view.py",
                "title": "Зображення DICOM",
                "subtitle": f"{info['description']}, оконне перетворення у шейдері",
                "info": info,
                "histogram": dicom_data.histogram(values),
                "controls": [
                    {
                        "id": "windowCenter",
                        "label": "WINDOW CENTER",
                        "unit": " HU",
                        "min": round(low),
                        "max": round(high),
                        "step": 5.0,
                        "value": round(info["defaultCenter"]),
                    },
                    {
                        "id": "windowWidth",
                        "label": "WINDOW WIDTH",
                        "unit": " HU",
                        "min": 40.0,
                        "max": round(span * 2),
                        "step": 10.0,
                        "value": round(info["defaultWidth"]),
                    },
                ],
                "presets": [
                    {"label": "brain 40/80", "params": {"windowCenter": 40, "windowWidth": 80}},
                    {"label": "bone 400/1800", "params": {"windowCenter": 400, "windowWidth": 1800}},
                    {"label": "soft 60/400", "params": {"windowCenter": 60, "windowWidth": 400}},
                ],
                "note": "Межі вікна показані вертикальними лініями: усе за ними "
                        "йде в чорне або біле.",
            },
        ],
    }
