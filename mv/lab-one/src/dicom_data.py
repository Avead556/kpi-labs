import numpy as np
import pydicom
from pydicom.data import get_testdata_file
from pydicom.pixels import apply_modality_lut

DEFAULT_FILE = "CT_small.dcm"


def resolve_path(path=None):
    return path or get_testdata_file(DEFAULT_FILE)


def load_dicom(path=None):
    dataset = pydicom.dcmread(resolve_path(path))
    pixels = dataset.pixel_array
    if pixels.ndim == 3 and "NumberOfFrames" in dataset:
        pixels = pixels[0]
    if pixels.ndim != 2:
        raise ValueError(f"expected a single-component image, got shape {pixels.shape}")

    values = np.ascontiguousarray(apply_modality_lut(pixels, dataset), dtype=np.float32)
    return values, dataset


def default_window(values, dataset):
    center = dataset.get("WindowCenter")
    width = dataset.get("WindowWidth")
    if center is not None and width is not None:
        center = float(center[0] if isinstance(center, pydicom.multival.MultiValue) else center)
        width = float(width[0] if isinstance(width, pydicom.multival.MultiValue) else width)
        if width >= 1.0:
            return center, width

    low, high = float(values.min()), float(values.max())
    return (low + high) / 2.0, max(high - low, 1.0)


def is_inverted(dataset):
    return dataset.get("PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1"


def describe(dataset, values):
    height, width = values.shape
    modality = dataset.get("Modality", "?")
    body_part = dataset.get("BodyPartExamined", "")
    return f"{modality} {width}x{height}" + (f" {body_part}" if body_part else "")


def metadata(dataset, values):
    height, width = values.shape
    center, width_window = default_window(values, dataset)
    return {
        "modality": str(dataset.get("Modality", "?")),
        "bodyPart": str(dataset.get("BodyPartExamined", "")),
        "rows": int(height),
        "columns": int(width),
        "minValue": float(values.min()),
        "maxValue": float(values.max()),
        "defaultCenter": float(center),
        "defaultWidth": float(width_window),
        "inverted": bool(is_inverted(dataset)),
        "description": describe(dataset, values),
    }


def histogram(values, bins=72):
    counts, edges = np.histogram(values, bins=bins)
    return {
        "bins": [int(count) for count in counts],
        "min": float(edges[0]),
        "max": float(edges[-1]),
    }
