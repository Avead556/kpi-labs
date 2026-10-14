import io
import urllib.request
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ENERGY_CSV = DATA_DIR / "energy_efficiency.csv"
ENERGY_URL = "https://archive.ics.uci.edu/static/public/242/energy+efficiency.zip"

ENERGY_FEATURES = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]
ENERGY_TARGETS = ["Y1", "Y2"]
ENERGY_LABELS = {
    "X1": "відносна компактність", "X2": "площа", "X3": "площа стін",
    "X4": "площа даху", "X5": "загальна висота", "X6": "орієнтація",
    "X7": "площа скління", "X8": "розподіл скління",
    "Y1": "навантаження на обігрів", "Y2": "навантаження на охолодження",
}

_SHEET = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _sheet_rows(xlsx_bytes):
    with zipfile.ZipFile(io.BytesIO(xlsx_bytes)) as book:
        shared = []
        if "xl/sharedStrings.xml" in book.namelist():
            root = ElementTree.fromstring(book.read("xl/sharedStrings.xml"))
            shared = ["".join(t.text or "" for t in item.iter(f"{_SHEET}t"))
                      for item in root.iter(f"{_SHEET}si")]
        sheet = ElementTree.fromstring(book.read("xl/worksheets/sheet1.xml"))

    for row in sheet.iter(f"{_SHEET}row"):
        values = {}
        for cell in row.iter(f"{_SHEET}c"):
            value = cell.find(f"{_SHEET}v")
            if value is None or value.text is None:
                continue
            column = "".join(ch for ch in cell.get("r") if ch.isalpha())
            values[column] = shared[int(value.text)] if cell.get("t") == "s" else value.text
        if values:
            yield values


def load_energy():
    if not ENERGY_CSV.exists():
        with urllib.request.urlopen(ENERGY_URL, timeout=60) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        header, *rows = _sheet_rows(archive.read("ENB2012_data.xlsx"))
        columns = sorted(header, key=lambda letters: (len(letters), letters))
        frame = pd.DataFrame([[row[c] for c in columns] for row in rows if len(row) == len(columns)],
                             columns=[header[c].upper() for c in columns]).astype(float)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        frame.to_csv(ENERGY_CSV, index=False)
    return pd.read_csv(ENERGY_CSV)


def load_diabetes():
    from sklearn import datasets

    bunch = datasets.load_diabetes(as_frame=True, scaled=True)
    frame = bunch.frame.rename(columns={"target": "progression"})
    return frame
