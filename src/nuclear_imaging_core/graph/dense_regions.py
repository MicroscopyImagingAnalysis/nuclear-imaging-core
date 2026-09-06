from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import feature, filters, segmentation
from skimage.measure import find_contours, label, regionprops_table
from skimage.morphology import remove_small_objects
from skimage.segmentation import relabel_sequential


@dataclass(frozen=True)
class DenseRegionConfig:
    # Supported modes: "quantile", "mean_std", "multiotsu".
    threshold_mode: str = "quantile"
    # Quantile is evaluated on intensities inside the nucleus mask only.
    threshold_quantile: float = 80.0
    # Used when threshold_mode == "mean_std".
    alpha: float = 1.0
    # Used when threshold_mode == "multiotsu". Must be <= 3.
    multiotsu_classes: int = 3
    with_peaks: bool = False
    peak_distance_scale: float = 3.0
    with_boundary_nodes: bool = False
    boundary_num_points: int = 16
    gaussian_sigma_px: float = 3.0
    min_region_size_px: int = 9
    connectivity: int = 1


@dataclass
class DenseFrameResult:
    frame_name: str
    normalized: np.ndarray
    blurred: np.ndarray
    nucleus_mask: np.ndarray
    nucleus_centroid_y: float
    nucleus_centroid_x: float
    threshold: float
    dense_binary: np.ndarray
    dense_labeled: np.ndarray
    peak_markers: np.ndarray
    peak_table: pd.DataFrame
    boundary_table: pd.DataFrame
    node_table: pd.DataFrame


def derive_nucleus_mask(frame: np.ndarray) -> np.ndarray:
    """Infer nucleus mask from crop convention: non-zero pixels are inside nucleus."""
    return np.asarray(frame > 0, dtype=bool)


def normalize_within_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(frame, dtype=np.float32)
    if np.count_nonzero(mask) == 0:
        return out

    vals = frame[mask].astype(np.float32)
    vmin = float(vals.min())
    vmax = float(vals.max())
    if vmax > vmin:
        out[mask] = (vals - vmin) / (vmax - vmin)
    else:
        out[mask] = 0.0
    return out


def threshold_dense_regions(
    normalized: np.ndarray,
    mask: np.ndarray,
    threshold_mode: str = "quantile",
    threshold_quantile: float = 80.0,
    alpha: float = 1.0,
    multiotsu_classes: int = 3,
) -> tuple[float, np.ndarray]:
    """
    Threshold dense regions within the nucleus mask using one of:
    - "quantile": thr = percentile(vals, threshold_quantile)
    - "mean_std": thr = mean(vals) + alpha * std(vals)
    - "multiotsu": thr = highest threshold from threshold_multiotsu(vals, classes=n)
    """
    vals = normalized[mask]
    if vals.size == 0:
        return 1.0, np.zeros_like(normalized, dtype=bool)

    mode = str(threshold_mode).strip().lower()
    dense = np.zeros_like(normalized, dtype=bool)

    if mode == "quantile":
        q = float(np.clip(threshold_quantile, 0.0, 100.0))
        thr = float(np.percentile(vals, q))
        # Define dense nodes using only masked intensities.
        dense_vals = vals >= thr
        dense[mask] = dense_vals
    elif mode == "mean_std":
        mu = float(vals.mean())
        sigma = float(vals.std())
        thr = float(np.clip(mu + float(alpha) * sigma, 0.0, 1.0))
        # Define dense nodes using only masked intensities.
        dense_vals = vals >= thr
        dense[mask] = dense_vals
    elif mode == "multiotsu":
        n = int(multiotsu_classes)
        if n > 3:
            raise ValueError("multiotsu_classes must be <= 3.")
        if n < 2:
            raise ValueError("multiotsu_classes must be >= 2.")
        unique_count = int(np.unique(vals).size)
        if unique_count < n:
            raise ValueError(
                f"multiotsu_classes={n} requires at least {n} distinct masked intensities, got {unique_count}."
            )
        thresholds = filters.threshold_multiotsu(vals, classes=n)
        thr = float(thresholds[-1])
        # Highest class from masked pixels only.
        classes = np.digitize(vals, bins=thresholds)
        dense_vals = classes == (n - 1)
        dense[mask] = dense_vals
    else:
        raise ValueError(
            f"Unknown threshold_mode={threshold_mode!r}. "
            "Expected one of {'quantile', 'mean_std', 'multiotsu'}."
        )
    return thr, dense


def blur_within_mask(image: np.ndarray, mask: np.ndarray, sigma_px: float) -> np.ndarray:
    """
    Gaussian blur image, while preserving outside-mask zeros.
    """
    if sigma_px <= 0:
        return image.copy()
    blurred = ndi.gaussian_filter(image.astype(np.float32), sigma=float(sigma_px))
    blurred[~mask] = 0.0
    return blurred


def nucleus_centroid(mask: np.ndarray) -> tuple[float, float]:
    if np.count_nonzero(mask) == 0:
        h, w = mask.shape
        return float(h / 2.0), float(w / 2.0)
    cy, cx = ndi.center_of_mass(mask.astype(np.uint8))
    return float(cy), float(cx)


def clean_and_label_dense(dense_binary: np.ndarray, min_region_size_px: int, connectivity: int) -> np.ndarray:
    if np.count_nonzero(dense_binary) == 0:
        return np.zeros_like(dense_binary, dtype=np.int32)

    cleaned = remove_small_objects(dense_binary, min_size=min_region_size_px)
    labeled = label(cleaned, connectivity=connectivity)
    return np.asarray(labeled, dtype=np.int32)


def peak_table_from_markers(markers: np.ndarray) -> pd.DataFrame:
    coords = np.argwhere(markers > 0)
    if coords.size == 0:
        return pd.DataFrame(columns=["peak_id", "peak_y", "peak_x"])
    peak_ids = markers[markers > 0].astype(int)
    out = pd.DataFrame(
        {
            "peak_id": peak_ids,
            "peak_y": coords[:, 0].astype(float),
            "peak_x": coords[:, 1].astype(float),
        }
    )
    return out.sort_values("peak_id").reset_index(drop=True)


def _signed_contour_area_xy(contour_xy: np.ndarray) -> float:
    if contour_xy.shape[0] < 3:
        return 0.0
    x = contour_xy[:, 0]
    y = contour_xy[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _choose_contour_start_index(contour_yx: np.ndarray, nuc_centroid_y: float, nuc_centroid_x: float) -> int:
    dy = contour_yx[:, 0] - float(nuc_centroid_y)
    dx = contour_yx[:, 1] - float(nuc_centroid_x)
    angles = np.arctan2(-dy, dx)
    wrapped = np.abs(np.angle(np.exp(1j * angles)))
    return int(np.argmin(wrapped))


def _resample_closed_contour(contour_yx: np.ndarray, num_points: int) -> np.ndarray:
    if contour_yx.shape[0] == 0 or num_points <= 0:
        return np.zeros((0, 2), dtype=np.float64)

    pts = contour_yx.astype(np.float64)
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total <= 1e-12:
        return np.repeat(pts[:1], repeats=num_points, axis=0)

    targets = np.linspace(0.0, total, num_points, endpoint=False)
    out = np.zeros((num_points, 2), dtype=np.float64)
    for i, t in enumerate(targets):
        idx = int(np.searchsorted(cum, t, side="right") - 1)
        idx = min(max(idx, 0), len(seg) - 1)
        seg_len = float(seg[idx])
        if seg_len <= 1e-12:
            out[i] = pts[idx]
            continue
        frac = (t - cum[idx]) / seg_len
        out[i] = pts[idx] + frac * (pts[idx + 1] - pts[idx])
    return out


def boundary_nodes_from_mask(
    mask: np.ndarray,
    nuc_centroid_y: float,
    nuc_centroid_x: float,
    num_points: int,
) -> pd.DataFrame:
    if np.count_nonzero(mask) == 0 or int(num_points) <= 0:
        return pd.DataFrame(
            columns=[
                "boundary_id",
                "boundary_order",
                "boundary_y",
                "boundary_x",
                "rel_boundary_y",
                "rel_boundary_x",
                "nucleus_centroid_y",
                "nucleus_centroid_x",
                "angle_rad",
            ]
        )

    contours = find_contours(mask.astype(np.uint8), level=0.5)
    if not contours:
        return pd.DataFrame(
            columns=[
                "boundary_id",
                "boundary_order",
                "boundary_y",
                "boundary_x",
                "rel_boundary_y",
                "rel_boundary_x",
                "nucleus_centroid_y",
                "nucleus_centroid_x",
                "angle_rad",
            ]
        )

    contour_yx = max(contours, key=lambda c: c.shape[0]).astype(np.float64)
    contour_xy = contour_yx[:, ::-1]
    if _signed_contour_area_xy(contour_xy) > 0:
        contour_yx = contour_yx[::-1]

    start_idx = _choose_contour_start_index(contour_yx, nuc_centroid_y, nuc_centroid_x)
    contour_yx = np.vstack([contour_yx[start_idx:], contour_yx[:start_idx]])
    sampled = _resample_closed_contour(contour_yx, int(num_points))

    rel_y = sampled[:, 0] - float(nuc_centroid_y)
    rel_x = sampled[:, 1] - float(nuc_centroid_x)
    angles = np.arctan2(-rel_y, rel_x)

    out = pd.DataFrame(
        {
            "boundary_id": np.arange(0, int(num_points), dtype=int),
            "boundary_order": np.arange(0, int(num_points), dtype=int),
            "boundary_y": sampled[:, 0],
            "boundary_x": sampled[:, 1],
            "rel_boundary_y": rel_y,
            "rel_boundary_x": rel_x,
            "nucleus_centroid_y": float(nuc_centroid_y),
            "nucleus_centroid_x": float(nuc_centroid_x),
            "angle_rad": angles,
        }
    )
    return out


def _peak_min_distance_px(min_region_size_px: int, peak_distance_scale: float) -> int:
    return max(1, round(float(peak_distance_scale) * np.sqrt(float(min_region_size_px))))


def _dense_peak_markers(
    image: np.ndarray,
    dense_binary: np.ndarray,
    min_region_size_px: int,
    peak_distance_scale: float,
    connectivity: int,
) -> np.ndarray:
    markers = np.zeros_like(dense_binary, dtype=np.int32)
    if np.count_nonzero(dense_binary) == 0:
        return markers

    min_distance = _peak_min_distance_px(min_region_size_px, peak_distance_scale)
    peak_coords = feature.peak_local_max(
        image,
        min_distance=min_distance,
        labels=dense_binary.astype(np.uint8),
        exclude_border=False,
    )

    dense_components = label(dense_binary, connectivity=connectivity)
    coord_list: list[tuple[int, int]] = [tuple(int(v) for v in xy) for xy in peak_coords.tolist()]
    existing = set(coord_list)

    # Guarantee at least one peak per dense connected component.
    for comp_id in range(1, int(dense_components.max()) + 1):
        comp_mask = dense_components == comp_id
        if not np.any(comp_mask):
            continue
        has_peak = any(comp_mask[y, x] for y, x in coord_list)
        if has_peak:
            continue
        comp_vals = np.where(comp_mask, image, -np.inf)
        flat_idx = int(np.argmax(comp_vals))
        py, px = np.unravel_index(flat_idx, image.shape)
        coord = (int(py), int(px))
        if coord not in existing:
            coord_list.append(coord)
            existing.add(coord)

    for peak_id, (py, px) in enumerate(coord_list, start=1):
        markers[int(py), int(px)] = int(peak_id)
    return markers


def split_dense_with_peaks(
    image: np.ndarray,
    dense_binary: np.ndarray,
    min_region_size_px: int,
    peak_distance_scale: float,
    connectivity: int,
) -> tuple[np.ndarray, np.ndarray]:
    if np.count_nonzero(dense_binary) == 0:
        empty = np.zeros_like(dense_binary, dtype=np.int32)
        return empty, empty

    peak_markers = _dense_peak_markers(
        image=image,
        dense_binary=dense_binary,
        min_region_size_px=min_region_size_px,
        peak_distance_scale=peak_distance_scale,
        connectivity=connectivity,
    )
    if int(peak_markers.max()) == 0:
        empty = np.zeros_like(dense_binary, dtype=np.int32)
        return empty, empty

    watershed_labels = segmentation.watershed(
        -image.astype(np.float32),
        markers=peak_markers,
        mask=dense_binary.astype(bool),
    )
    if int(np.max(watershed_labels)) <= 1:
        cleaned_bool = remove_small_objects(watershed_labels.astype(bool), min_size=min_region_size_px)
        cleaned = watershed_labels * cleaned_bool.astype(watershed_labels.dtype)
    else:
        cleaned = remove_small_objects(watershed_labels, min_size=min_region_size_px)
    relabeled, _, _ = relabel_sequential(cleaned)
    return np.asarray(relabeled, dtype=np.int32), np.asarray(peak_markers, dtype=np.int32)


def nodes_from_labeled_dense(
    labeled_dense: np.ndarray,
    nuc_centroid_y: float,
    nuc_centroid_x: float,
) -> pd.DataFrame:
    if int(labeled_dense.max()) == 0:
        return pd.DataFrame(
            columns=[
                "node_id",
                "dense_label",
                "size_px",
                "centroid_y",
                "centroid_x",
                "rel_centroid_y",
                "rel_centroid_x",
                "nucleus_centroid_y",
                "nucleus_centroid_x",
            ]
        )

    tbl = regionprops_table(
        labeled_dense,
        properties=("label", "area", "centroid"),
    )
    df = pd.DataFrame(tbl).rename(
        columns={
            "label": "dense_label",
            "area": "size_px",
            "centroid-0": "centroid_y",
            "centroid-1": "centroid_x",
        }
    )
    df["dense_label"] = df["dense_label"].astype(int)
    # Keep external node IDs aligned with dense labels (1-based labels, 0 is background).
    df["node_id"] = df["dense_label"].astype(int)
    df["size_px"] = df["size_px"].astype(int)
    df["nucleus_centroid_y"] = float(nuc_centroid_y)
    df["nucleus_centroid_x"] = float(nuc_centroid_x)
    df["rel_centroid_y"] = df["centroid_y"] - float(nuc_centroid_y)
    df["rel_centroid_x"] = df["centroid_x"] - float(nuc_centroid_x)
    return df.sort_values("node_id").reset_index(drop=True)


def segment_dense_regions_frame(frame: np.ndarray, frame_name: str, config: DenseRegionConfig) -> DenseFrameResult:
    mask = derive_nucleus_mask(frame)
    normalized = normalize_within_mask(frame, mask)
    blurred = blur_within_mask(normalized, mask, sigma_px=config.gaussian_sigma_px)
    nuc_cy, nuc_cx = nucleus_centroid(mask)
    thr, dense_binary = threshold_dense_regions(
        blurred,
        mask,
        threshold_mode=config.threshold_mode,
        threshold_quantile=config.threshold_quantile,
        alpha=config.alpha,
        multiotsu_classes=config.multiotsu_classes,
    )
    if config.with_peaks:
        labeled, peak_markers = split_dense_with_peaks(
            image=blurred,
            dense_binary=dense_binary,
            min_region_size_px=config.min_region_size_px,
            peak_distance_scale=config.peak_distance_scale,
            connectivity=config.connectivity,
        )
    else:
        labeled = clean_and_label_dense(
            dense_binary,
            min_region_size_px=config.min_region_size_px,
            connectivity=config.connectivity,
        )
        peak_markers = np.zeros_like(dense_binary, dtype=np.int32)
    nodes = nodes_from_labeled_dense(labeled, nuc_cy, nuc_cx)
    nodes.insert(0, "frame", frame_name)
    peak_table = peak_table_from_markers(peak_markers)
    boundary_table = boundary_nodes_from_mask(
        mask=mask,
        nuc_centroid_y=nuc_cy,
        nuc_centroid_x=nuc_cx,
        num_points=config.boundary_num_points if config.with_boundary_nodes else 0,
    )
    if not boundary_table.empty:
        boundary_table.insert(0, "frame", frame_name)

    return DenseFrameResult(
        frame_name=frame_name,
        normalized=normalized,
        blurred=blurred,
        nucleus_mask=mask,
        nucleus_centroid_y=nuc_cy,
        nucleus_centroid_x=nuc_cx,
        threshold=thr,
        dense_binary=dense_binary,
        dense_labeled=labeled,
        peak_markers=peak_markers,
        peak_table=peak_table,
        boundary_table=boundary_table,
        node_table=nodes,
    )


def segment_dense_regions_triplet(
    triplet_frames: Dict[str, np.ndarray],
    config: DenseRegionConfig,
    frame_order: tuple[str, str, str] = ("ref", "dec", "fin"),
) -> Dict[str, DenseFrameResult]:
    out: Dict[str, DenseFrameResult] = {}
    for frame_name in frame_order:
        out[frame_name] = segment_dense_regions_frame(triplet_frames[frame_name], frame_name, config)
    return out
