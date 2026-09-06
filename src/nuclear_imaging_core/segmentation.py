from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy import ndimage as ndi
from skimage import exposure, filters, measure, morphology, segmentation

AnalysisMode = Literal["2D", "3D"]
SegmentationMethod = Literal["stardist2d", "stardist2d_stack", "multiotsu", "multiotsu_simple"]


@dataclass(frozen=True)
class SegmentationConfig:
    analysis_mode: AnalysisMode = "2D"
    segmentation_method: SegmentationMethod = "stardist2d"
    stardist_model_name: str = "2D_versatile_fluo"
    enhance: bool = False
    blocksize: int = 1024
    gamma: float = 0.7
    min_size: int = 50
    big_image_size_threshold: int = 2000


def _require_stardist():
    try:
        from stardist.models import StarDist2D  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "stardist is required for baseline segmentation. Install stardist/csbdeep before running."
        ) from exc
    return StarDist2D


def load_stardist_model(model_name: str):
    StarDist2D = _require_stardist()
    local_basedir = Path.home() / ".keras" / "models" / "StarDist2D"
    local_model_dir = local_basedir / str(model_name)
    if local_model_dir.exists():
        return StarDist2D(None, name=str(model_name), basedir=str(local_basedir))
    return StarDist2D.from_pretrained(model_name)


def normalize_image(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    amin = float(np.min(arr))
    amax = float(np.max(arr))
    denom = max(amax - amin, 1e-8)
    return (arr - amin) / denom


def segment2d_stardist(
    img: np.ndarray,
    model,
    *,
    enhance: bool = False,
    blocksize: int = 1024,
    gamma: float = 0.7,
    min_size: int = 50,
    big_image_size_threshold: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.zeros_like(img, dtype=np.uint16)
    img_norm = normalize_image(img)

    if not enhance:
        if img_norm.shape[0] > int(big_image_size_threshold):
            labels, _ = model.predict_instances_big(
                img_norm,
                axes="YX",
                block_size=int(blocksize),
                min_overlap=200,
                context=200,
                show_progress=False,
                labels_out_dtype=np.uint16,
            )
        else:
            labels, _ = model.predict_instances(img_norm)
    else:
        proc_img_norm = exposure.adjust_gamma(img_norm, gamma)
        threshold = filters.threshold_otsu(proc_img_norm)
        labels = proc_img_norm > threshold
        labels = ndi.binary_fill_holes(labels)
        labels = morphology.binary_opening(labels)
        labels = morphology.remove_small_objects(labels, min_size=min_size)
        labels = segmentation.clear_border(labels, buffer_size=1)
        labels = measure.label(labels)

    labels = np.asarray(labels)
    for label_id in np.unique(labels[labels > 0]):
        if int(np.count_nonzero(labels == label_id)) < int(2 * min_size):
            labels[labels == label_id] = 0
    labels = measure.label(labels > 0).astype(np.uint16)
    return labels, labels > 0


def brushfire_combine_stacks(label_stack: np.ndarray) -> np.ndarray:
    """Merge slice-wise labels into a volume by propagating overlap-consistent IDs."""
    stack = np.asarray(label_stack)
    if stack.ndim != 3:
        raise ValueError(f"Expected (Z,Y,X) label stack, got {stack.shape}")
    if stack.shape[0] == 0:
        return stack.astype(np.uint16)

    relabelled = np.zeros_like(stack, dtype=np.uint16)
    unique_counts = [len(np.unique(stack[z][stack[z] > 0])) for z in range(stack.shape[0])]
    z_mid = int(np.argmax(unique_counts)) if stack.shape[0] >= 1 else 0
    relabelled[z_mid] = measure.label(stack[z_mid] > 0).astype(np.uint16)
    next_label = int(relabelled[z_mid].max())

    order: list[tuple[int, int]] = []
    for offset in range(1, stack.shape[0]):
        z_up = z_mid + offset
        z_down = z_mid - offset
        if z_up < stack.shape[0]:
            order.append((z_up - 1, z_up))
        if z_down >= 0:
            order.append((z_down + 1, z_down))

    for prev_idx, curr_idx in order:
        prev_slice = relabelled[prev_idx]
        curr_slice = measure.label(stack[curr_idx] > 0).astype(np.uint16)
        curr_out = np.zeros_like(curr_slice, dtype=np.uint16)
        for curr_label in np.unique(curr_slice[curr_slice > 0]):
            curr_mask = curr_slice == curr_label
            overlap_ids, counts = np.unique(prev_slice[curr_mask], return_counts=True)
            valid = overlap_ids > 0
            overlap_ids = overlap_ids[valid]
            counts = counts[valid]
            if overlap_ids.size > 0:
                curr_out[curr_mask] = int(overlap_ids[np.argmax(counts)])
            else:
                next_label += 1
                curr_out[curr_mask] = next_label
        relabelled[curr_idx] = curr_out

    return measure.label(relabelled > 0, connectivity=1).astype(np.uint16) if np.max(relabelled) == 1 else relabelled


def segment3d_stardist_stack(
    img: np.ndarray,
    model,
    *,
    enhance: bool = False,
    blocksize: int = 1024,
    gamma: float = 0.7,
    min_size: int = 100,
    big_image_size_threshold: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(img)
    if arr.ndim != 3:
        raise ValueError(f"Expected (Z,Y,X) image for 3D segmentation, got {arr.shape}")
    slice_labels = np.zeros_like(arr, dtype=np.uint16)
    for z in range(arr.shape[0]):
        labels2d, _ = segment2d_stardist(
            arr[z],
            model,
            enhance=enhance,
            blocksize=blocksize,
            gamma=gamma,
            min_size=min_size,
            big_image_size_threshold=big_image_size_threshold,
        )
        slice_labels[z] = segmentation.clear_border(labels2d, buffer_size=1)
    relabelled = brushfire_combine_stacks(slice_labels)
    relabelled = measure.label(relabelled > 0, connectivity=1).astype(np.uint16)
    return relabelled, relabelled > 0


def segment3d_multiotsu_simple(img: np.ndarray, *, min_size: int = 100) -> tuple[np.ndarray, np.ndarray]:
    arr = normalize_image(img)
    threshold = filters.threshold_multiotsu(arr, classes=3)[1]
    mask = arr > threshold
    mask = ndi.binary_fill_holes(mask)
    mask = morphology.remove_small_objects(mask, min_size=min_size)
    labels = measure.label(mask, connectivity=1).astype(np.uint16)
    return labels, labels > 0


def segment3d_multiotsu(img: np.ndarray, *, min_size: int = 100) -> tuple[np.ndarray, np.ndarray]:
    arr = normalize_image(img)
    labels = np.zeros_like(arr, dtype=np.uint16)
    for z in range(arr.shape[0]):
        threshold = filters.threshold_multiotsu(arr[z], classes=3)[1]
        mask = arr[z] > threshold
        mask = segmentation.clear_border(mask, buffer_size=1)
        mask = morphology.remove_small_objects(mask, min_size=min_size)
        labels[z] = measure.label(mask).astype(np.uint16)
    relabelled = brushfire_combine_stacks(labels)
    relabelled = measure.label(relabelled > 0, connectivity=1).astype(np.uint16)
    return relabelled, relabelled > 0


def segment_frame(
    img: np.ndarray,
    cfg: SegmentationConfig,
    *,
    model=None,
) -> tuple[np.ndarray, np.ndarray]:
    mode = str(cfg.analysis_mode).upper()
    method = str(cfg.segmentation_method).lower()
    if mode == "2D":
        if method != "stardist2d":
            raise ValueError(f"2D mode currently supports segmentation_method='stardist2d', got {cfg.segmentation_method!r}")
        if model is None:
            model = load_stardist_model(cfg.stardist_model_name)
        return segment2d_stardist(
            img,
            model,
            enhance=cfg.enhance,
            blocksize=cfg.blocksize,
            gamma=cfg.gamma,
            min_size=cfg.min_size,
            big_image_size_threshold=cfg.big_image_size_threshold,
        )

    if mode != "3D":
        raise ValueError(f"Unsupported analysis_mode={cfg.analysis_mode!r}")
    if method == "stardist2d_stack":
        if model is None:
            model = load_stardist_model(cfg.stardist_model_name)
        return segment3d_stardist_stack(
            img,
            model,
            enhance=cfg.enhance,
            blocksize=cfg.blocksize,
            gamma=cfg.gamma,
            min_size=cfg.min_size,
            big_image_size_threshold=cfg.big_image_size_threshold,
        )
    if method == "multiotsu":
        return segment3d_multiotsu(img, min_size=cfg.min_size)
    if method == "multiotsu_simple":
        return segment3d_multiotsu_simple(img, min_size=cfg.min_size)
    raise ValueError(f"Unsupported 3D segmentation method: {cfg.segmentation_method!r}")


__all__ = [
    "AnalysisMode",
    "SegmentationMethod",
    "SegmentationConfig",
    "brushfire_combine_stacks",
    "load_stardist_model",
    "normalize_image",
    "segment2d_stardist",
    "segment3d_multiotsu",
    "segment3d_multiotsu_simple",
    "segment3d_stardist_stack",
    "segment_frame",
]
