from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List, Sequence

import numpy as np
import pandas as pd

from .io import FRAME_NAMES, load_frame_crop


@dataclass(frozen=True)
class QCConfig:
    min_masked_area_px: int = 100
    saturation_value: int = 4093
    max_saturation_fraction: float = 0.10
    require_all_frames_pass: bool = True


@dataclass(frozen=True)
class FrameQC:
    masked_area_px: int
    saturated_area_px: int
    saturation_fraction: float
    pass_masked_area: bool
    pass_saturation: bool
    pass_frame: bool


@dataclass(frozen=True)
class TripletQCResult:
    pass_triplet: bool
    per_frame: Dict[str, FrameQC]


def compute_frame_qc(frame: np.ndarray, config: QCConfig) -> FrameQC:
    masked_area = int(np.count_nonzero(frame > 0))
    saturated_area = int(np.count_nonzero(frame >= config.saturation_value))
    sat_frac = float(saturated_area / (masked_area + 1e-8))

    pass_mask = masked_area >= config.min_masked_area_px
    pass_sat = sat_frac < config.max_saturation_fraction
    pass_frame = bool(pass_mask and pass_sat)

    return FrameQC(
        masked_area_px=masked_area,
        saturated_area_px=saturated_area,
        saturation_fraction=sat_frac,
        pass_masked_area=pass_mask,
        pass_saturation=pass_sat,
        pass_frame=pass_frame,
    )


def evaluate_triplet_qc(triplet_frames: Dict[str, np.ndarray], config: QCConfig) -> TripletQCResult:
    per_frame = {name: compute_frame_qc(img, config) for name, img in triplet_frames.items()}

    frame_passes = [v.pass_frame for v in per_frame.values()]
    if config.require_all_frames_pass:
        pass_triplet = bool(all(frame_passes))
    else:
        pass_triplet = bool(any(frame_passes))

    return TripletQCResult(pass_triplet=pass_triplet, per_frame=per_frame)


def qc_result_to_row(qc: TripletQCResult) -> dict:
    row = {"pass_triplet": qc.pass_triplet}
    for frame_name, fr in qc.per_frame.items():
        row[f"{frame_name}_masked_area_px"] = fr.masked_area_px
        row[f"{frame_name}_saturated_area_px"] = fr.saturated_area_px
        row[f"{frame_name}_saturation_fraction"] = fr.saturation_fraction
        row[f"{frame_name}_pass_masked_area"] = fr.pass_masked_area
        row[f"{frame_name}_pass_saturation"] = fr.pass_saturation
        row[f"{frame_name}_pass_frame"] = fr.pass_frame
    return row


def filter_records_by_qc(
    records_df: pd.DataFrame,
    config: QCConfig,
    *,
    frame_order: Sequence[str | int] = FRAME_NAMES,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate QC on each record and return:
    - kept records dataframe
    - full qc dataframe joined with triplet identifiers
    """
    qc_rows: List[dict] = []

    for r in records_df.itertuples(index=False):
        frames = load_frame_crop(r.crop_path, frame_order=frame_order)
        qc = evaluate_triplet_qc(frames, config)
        row = {
            "experiment_id": str(r.experiment_id),
            "nd2_prefix": str(r.nd2_prefix),
            "crop_id": int(r.crop_id),
            "crop_path": str(r.crop_path),
        }
        row.update(qc_result_to_row(qc))
        qc_rows.append(row)

    qc_df = pd.DataFrame(qc_rows)
    if qc_df.empty:
        return records_df.iloc[0:0].copy(), qc_df

    kept = qc_df[qc_df["pass_triplet"]].copy()
    kept_records = kept[["experiment_id", "nd2_prefix", "crop_id", "crop_path"]].reset_index(drop=True)
    return kept_records, qc_df.reset_index(drop=True)
