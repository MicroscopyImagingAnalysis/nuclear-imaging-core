"""Concise measurements for labelled 2D and 3D microscopy images."""

from __future__ import annotations

import numpy as np
import pandas as pd
from skimage import measure


def describe_image(image: np.ndarray) -> dict[str, object]:
    """Return dimensions, data type, range and robust intensity percentiles."""
    array = np.asarray(image)
    finite = array[np.isfinite(array)]
    percentiles = np.percentile(finite, [1, 50, 99]) if finite.size else [np.nan] * 3
    return {
        "shape": tuple(int(value) for value in array.shape),
        "ndim": int(array.ndim),
        "dtype": str(array.dtype),
        "minimum": float(np.min(finite)) if finite.size else np.nan,
        "p01": float(percentiles[0]),
        "median": float(percentiles[1]),
        "p99": float(percentiles[2]),
        "maximum": float(np.max(finite)) if finite.size else np.nan,
    }


def region_feature_table(image: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Measure labelled regions while preserving pixel-based geometry."""
    intensity = np.asarray(image)
    labelled = np.asarray(labels)
    if intensity.shape != labelled.shape:
        raise ValueError("image and labels must have the same shape")
    properties = [
        "label",
        "area",
        "centroid",
        "bbox",
        "intensity_mean",
        "intensity_min",
        "intensity_max",
    ]
    if labelled.ndim == 2:
        properties.extend(["eccentricity", "major_axis_length", "minor_axis_length", "solidity"])
    table = measure.regionprops_table(labelled, intensity_image=intensity, properties=properties)
    return pd.DataFrame(table).sort_values("label").reset_index(drop=True)


def normalized_radial_profile(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    bins: int = 10,
) -> pd.DataFrame:
    """Summarize intensity from object centre to boundary in normalized shells."""
    intensity = np.asarray(image, dtype=float)
    binary = np.asarray(mask, dtype=bool)
    if intensity.shape != binary.shape:
        raise ValueError("image and mask must have the same shape")
    if bins < 1:
        raise ValueError("bins must be positive")
    coordinates = np.argwhere(binary)
    if not coordinates.size:
        return pd.DataFrame(columns=["shell", "radius_start", "radius_end", "mean_intensity", "pixel_count"])
    centre = coordinates.mean(axis=0)
    distances = np.sqrt(np.sum((coordinates - centre) ** 2, axis=1))
    maximum = float(distances.max()) or 1.0
    normalized = distances / maximum
    edges = np.linspace(0.0, 1.0, bins + 1)
    values = intensity[binary]
    rows = []
    for shell in range(bins):
        selected = (normalized >= edges[shell]) & (
            normalized <= edges[shell + 1] if shell == bins - 1 else normalized < edges[shell + 1]
        )
        rows.append(
            {
                "shell": shell,
                "radius_start": float(edges[shell]),
                "radius_end": float(edges[shell + 1]),
                "mean_intensity": float(np.mean(values[selected])) if np.any(selected) else np.nan,
                "pixel_count": int(np.count_nonzero(selected)),
            }
        )
    return pd.DataFrame(rows)


__all__ = ["describe_image", "normalized_radial_profile", "region_feature_table"]
