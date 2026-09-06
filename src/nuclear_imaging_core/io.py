from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
from tifffile import imread

from .metadata import parse_experiment_id
from .timepoints import canonicalize_frame_order, default_frame_order


FRAME_NAMES = default_frame_order(3)


@dataclass(frozen=True)
class TripletRecord:
    experiment_id: str
    nd2_prefix: str
    crop_id: int
    crop_path: str

    def to_dict(self) -> dict:
        return asdict(self)


def _is_numeric_stem(path: Path) -> bool:
    return path.stem.isdigit()


def discover_triplet_records(crop_root: str | Path) -> pd.DataFrame:
    """
    Discover stack-style crop triplets from crop root.

    Expected path structure from batch pipeline:
      <crop_root>/<experiment_id>/<nd2_prefix>/<crop_id>.tif

    Only files with numeric stem are treated as crop triplets.
    Returned ``crop_id`` is 0-indexed within each ``(experiment_id, nd2_prefix)``
    group; original numeric stem is preserved in ``crop_id_raw``.
    """
    root = Path(crop_root)
    rows: List[dict] = []

    for p in sorted(root.rglob("*.tif")):
        if not _is_numeric_stem(p):
            continue
        parts = p.parts
        if len(parts) < 3:
            continue
        nd2_prefix = p.parent.name
        experiment_id = p.parent.parent.name
        rows.append(
            TripletRecord(
                experiment_id=experiment_id,
                nd2_prefix=nd2_prefix,
                crop_id=int(p.stem),
                crop_path=str(p),
            ).to_dict()
        )

    if not rows:
        return pd.DataFrame(
            columns=["experiment_id", "cell", "condition", "batch", "nd2_prefix", "crop_id", "crop_path"]
        )

    df = pd.DataFrame(rows)
    df["crop_id_raw"] = df["crop_id"].astype(int)
    # Normalize to 0-index within each (experiment_id, nd2_prefix).
    df["crop_id"] = (
        df.sort_values("crop_id_raw")
        .groupby(["experiment_id", "nd2_prefix"])["crop_id_raw"]
        .rank(method="dense")
        .astype(int)
        - 1
    )
    meta_df = df["experiment_id"].apply(parse_experiment_id).apply(pd.Series)
    df = pd.concat([df, meta_df], axis=1)
    df = df.sort_values(["experiment_id", "nd2_prefix", "crop_id"]).reset_index(drop=True)
    return df


def load_frame_stack(
    crop_path: str | Path,
    *,
    frame_order: Sequence[str | int] | None = None,
    n_frames: int | None = None,
) -> np.ndarray:
    """
    Load a crop stack as (T,H,W) or (T,Z,Y,X).

    The 4D form is the newer baseline extension; existing 2D graph crops retain
    their original 3D stack behavior.
    """
    arr = np.asarray(imread(str(crop_path)))
    if frame_order is None and n_frames is None:
        n_frames = len(FRAME_NAMES)
    resolved_order = canonicalize_frame_order(frame_order, n_frames=n_frames)
    if arr.ndim not in (3, 4) or arr.shape[0] < len(resolved_order):
        raise ValueError(
            f"Expected stack with shape (>={len(resolved_order)},H,W) or "
            f"(>={len(resolved_order)},Z,Y,X), got {arr.shape} for {crop_path}"
        )
    return arr[: len(resolved_order)]


def load_frame_crop(
    crop_path: str | Path,
    *,
    frame_order: Sequence[str | int] | None = None,
    n_frames: int | None = None,
) -> dict[str, np.ndarray]:
    resolved_order = canonicalize_frame_order(frame_order, n_frames=n_frames or len(FRAME_NAMES))
    stack = load_frame_stack(crop_path, frame_order=resolved_order)
    return {name: stack[i] for i, name in enumerate(resolved_order)}


def load_triplet_stack(crop_path: str | Path) -> np.ndarray:
    return load_frame_stack(crop_path, frame_order=FRAME_NAMES)


def load_triplet_crop(crop_path: str | Path) -> dict:
    """Load triplet stack into named frame dictionary: {'ref','dec','fin'} -> (H,W)."""
    return load_frame_crop(crop_path, frame_order=FRAME_NAMES)


def derive_mask_path(crop_path: str | Path) -> Path:
    path = Path(crop_path)
    return path.with_name(f"{path.stem}_mask{path.suffix}")


def load_frame_mask(
    crop_path: str | Path,
    *,
    frame_order: Sequence[str | int] | None = None,
    n_frames: int | None = None,
) -> dict[str, np.ndarray]:
    resolved_order = canonicalize_frame_order(frame_order, n_frames=n_frames or len(FRAME_NAMES))
    stack = load_frame_stack(derive_mask_path(crop_path), frame_order=resolved_order)
    return {name: stack[index].astype(bool) for index, name in enumerate(resolved_order)}


def load_triplet_mask(crop_path: str | Path) -> dict[str, np.ndarray]:
    return load_frame_mask(crop_path, frame_order=FRAME_NAMES)


def records_to_list(records_df: pd.DataFrame) -> List[TripletRecord]:
    return [
        TripletRecord(
            experiment_id=str(r.experiment_id),
            nd2_prefix=str(r.nd2_prefix),
            crop_id=int(r.crop_id),
            crop_path=str(r.crop_path),
        )
        for r in records_df.itertuples(index=False)
    ]


def subset_records(
    records_df: pd.DataFrame,
    experiment_id: str | None = None,
    nd2_prefix: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    out = records_df.copy()
    if experiment_id is not None:
        out = out[out["experiment_id"] == experiment_id]
    if nd2_prefix is not None:
        out = out[out["nd2_prefix"] == nd2_prefix]
    out = out.sort_values(["experiment_id", "nd2_prefix", "crop_id"]).reset_index(drop=True)
    if limit is not None:
        out = out.head(limit)
    return out


def iter_triplets(records_df: pd.DataFrame) -> Iterable[tuple[int, pd.Series]]:
    for idx, row in records_df.iterrows():
        yield idx, row


__all__ = [
    "FRAME_NAMES",
    "TripletRecord",
    "derive_mask_path",
    "discover_triplet_records",
    "iter_triplets",
    "load_frame_crop",
    "load_frame_mask",
    "load_frame_stack",
    "load_triplet_crop",
    "load_triplet_mask",
    "load_triplet_stack",
    "records_to_list",
    "subset_records",
]
