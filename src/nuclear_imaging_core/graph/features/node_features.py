from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..dense_regions import DenseFrameResult


def mask_max_radial_distance(mask: np.ndarray, cy: float, cx: float) -> float:
    if mask is None or np.count_nonzero(mask) == 0:
        return 1.0
    ys, xs = np.nonzero(mask)
    r = np.hypot(ys.astype(np.float64) - float(cy), xs.astype(np.float64) - float(cx))
    if r.size == 0:
        return 1.0
    rmax = float(np.max(r))
    return rmax if rmax > 1e-12 else 1.0


def _normalized_coords(rel_y: np.ndarray, rel_x: np.ndarray, rmax: float) -> tuple[np.ndarray, np.ndarray]:
    denom = max(float(rmax), 1e-12)
    return rel_y / denom, rel_x / denom


def _peak_intensity_stats(
    labeled_dense: np.ndarray,
    normalized: np.ndarray,
    peak_ids: np.ndarray,
    dense_labels: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for peak_id, dense_label in zip(peak_ids.astype(int), dense_labels.astype(int)):
        vals = normalized[labeled_dense == int(dense_label)]
        if vals.size == 0:
            rows.append(
                {
                    "peak_id": int(peak_id),
                    "intensity_min": np.nan,
                    "intensity_max": np.nan,
                    "intensity_mean": np.nan,
                    "intensity_median": np.nan,
                    "intensity_std": np.nan,
                }
            )
            continue
        rows.append(
            {
                "peak_id": int(peak_id),
                "intensity_min": float(np.min(vals)),
                "intensity_max": float(np.max(vals)),
                "intensity_mean": float(np.mean(vals)),
                "intensity_median": float(np.median(vals)),
                "intensity_std": float(np.std(vals, ddof=0)),
            }
        )
    return pd.DataFrame(rows)


def build_peak_node_table(frame_result: DenseFrameResult) -> pd.DataFrame:
    base = frame_result.node_table.copy()
    if base.empty:
        return pd.DataFrame(
            columns=[
                "graph_node_id",
                "node_type",
                "peak_id",
                "dense_label",
                "boundary_id",
                "boundary_order",
                "coord_y",
                "coord_x",
                "centroid_y",
                "centroid_x",
                "rel_y",
                "rel_x",
                "norm_y",
                "norm_x",
                "radial_distance_px",
                "radial_distance_norm",
                "size_px",
                "intensity_min",
                "intensity_max",
                "intensity_mean",
                "intensity_median",
                "intensity_std",
                "signed_angle_deg",
                "dist_prev_boundary",
                "dist_next_boundary",
                "dist_to_nearest_peak",
                "nucleus_centroid_y",
                "nucleus_centroid_x",
            ]
        )

    rmax = mask_max_radial_distance(
        frame_result.nucleus_mask,
        frame_result.nucleus_centroid_y,
        frame_result.nucleus_centroid_x,
    )
    rel_y = base["rel_centroid_y"].to_numpy(dtype=np.float64)
    rel_x = base["rel_centroid_x"].to_numpy(dtype=np.float64)
    norm_y, norm_x = _normalized_coords(rel_y, rel_x, rmax)
    radial_px = np.hypot(rel_y, rel_x)
    radial_norm = radial_px / max(rmax, 1e-12)

    stats = _peak_intensity_stats(
        labeled_dense=frame_result.dense_labeled,
        normalized=frame_result.normalized,
        peak_ids=base["node_id"].to_numpy(dtype=int),
        dense_labels=base.get("dense_label", base["node_id"]).to_numpy(dtype=int),
    )
    out = base.rename(
        columns={
            "node_id": "peak_id",
            "centroid_y": "coord_y",
            "centroid_x": "coord_x",
            "rel_centroid_y": "rel_y",
            "rel_centroid_x": "rel_x",
        }
    ).copy()
    out = out.merge(stats, on="peak_id", how="left")
    out["graph_node_id"] = out["peak_id"].astype(int)
    out["node_type"] = 0
    out["dense_label"] = out.get("dense_label", np.nan)
    out["boundary_id"] = -1
    out["boundary_order"] = -1
    out["centroid_y"] = out["coord_y"]
    out["centroid_x"] = out["coord_x"]
    out["norm_y"] = norm_y
    out["norm_x"] = norm_x
    out["radial_distance_px"] = radial_px
    out["radial_distance_norm"] = radial_norm
    out["signed_angle_deg"] = np.nan
    out["dist_prev_boundary"] = np.nan
    out["dist_next_boundary"] = np.nan
    out["dist_to_nearest_peak"] = np.nan
    keep_cols = [
        "frame",
        "graph_node_id",
        "node_type",
        "peak_id",
        "dense_label",
        "boundary_id",
        "boundary_order",
        "coord_y",
        "coord_x",
        "centroid_y",
        "centroid_x",
        "rel_y",
        "rel_x",
        "norm_y",
        "norm_x",
        "radial_distance_px",
        "radial_distance_norm",
        "size_px",
        "intensity_min",
        "intensity_max",
        "intensity_mean",
        "intensity_median",
        "intensity_std",
        "signed_angle_deg",
        "dist_prev_boundary",
        "dist_next_boundary",
        "dist_to_nearest_peak",
        "nucleus_centroid_y",
        "nucleus_centroid_x",
    ]
    return out[keep_cols].sort_values("graph_node_id").reset_index(drop=True)


def _signed_turning_angle_deg(coords_yx: np.ndarray) -> np.ndarray:
    n = coords_yx.shape[0]
    if n < 3:
        return np.zeros(n, dtype=np.float64)
    angles = np.zeros(n, dtype=np.float64)
    for i in range(n):
        prev_pt = coords_yx[(i - 1) % n]
        cur_pt = coords_yx[i]
        next_pt = coords_yx[(i + 1) % n]
        v_in = cur_pt - prev_pt
        v_out = next_pt - cur_pt
        cross = (v_in[1] * v_out[0]) - (v_in[0] * v_out[1])
        dot = float(np.dot(v_in, v_out))
        turn_deg = math.degrees(math.atan2(cross, dot))
        angles[i] = -turn_deg
    return angles


def build_boundary_node_table(frame_result: DenseFrameResult, peak_nodes: pd.DataFrame) -> pd.DataFrame:
    base = frame_result.boundary_table.copy()
    if base.empty:
        return pd.DataFrame(
            columns=[
                "graph_node_id",
                "node_type",
                "peak_id",
                "dense_label",
                "boundary_id",
                "boundary_order",
                "coord_y",
                "coord_x",
                "centroid_y",
                "centroid_x",
                "rel_y",
                "rel_x",
                "norm_y",
                "norm_x",
                "radial_distance_px",
                "radial_distance_norm",
                "size_px",
                "intensity_min",
                "intensity_max",
                "intensity_mean",
                "intensity_median",
                "intensity_std",
                "signed_angle_deg",
                "dist_prev_boundary",
                "dist_next_boundary",
                "dist_to_nearest_peak",
                "nucleus_centroid_y",
                "nucleus_centroid_x",
            ]
        )

    rmax = mask_max_radial_distance(
        frame_result.nucleus_mask,
        frame_result.nucleus_centroid_y,
        frame_result.nucleus_centroid_x,
    )
    rel_y = base["rel_boundary_y"].to_numpy(dtype=np.float64)
    rel_x = base["rel_boundary_x"].to_numpy(dtype=np.float64)
    norm_y, norm_x = _normalized_coords(rel_y, rel_x, rmax)
    radial_px = np.hypot(rel_y, rel_x)
    radial_norm = radial_px / max(rmax, 1e-12)

    coords_yx = base[["boundary_y", "boundary_x"]].to_numpy(dtype=np.float64)
    signed_angle_deg = _signed_turning_angle_deg(coords_yx)

    norm_coords_yx = np.column_stack([norm_y, norm_x])
    prev_coords = np.roll(norm_coords_yx, 1, axis=0)
    next_coords = np.roll(norm_coords_yx, -1, axis=0)
    dist_prev = np.linalg.norm(norm_coords_yx - prev_coords, axis=1)
    dist_next = np.linalg.norm(next_coords - norm_coords_yx, axis=1)

    if peak_nodes is not None and not peak_nodes.empty:
        peak_coords = peak_nodes[["norm_y", "norm_x"]].to_numpy(dtype=np.float64)
        dist_to_peak = np.min(
            np.linalg.norm(norm_coords_yx[:, None, :] - peak_coords[None, :, :], axis=2),
            axis=1,
        )
        max_peak_id = int(peak_nodes["peak_id"].max())
        peak_count = max_peak_id + 1
    else:
        dist_to_peak = np.full(len(base), np.nan, dtype=np.float64)
        peak_count = 0

    out = base.rename(
        columns={
            "boundary_y": "coord_y",
            "boundary_x": "coord_x",
            "rel_boundary_y": "rel_y",
            "rel_boundary_x": "rel_x",
        }
    ).copy()
    out["graph_node_id"] = peak_count + out["boundary_id"].astype(int)
    out["node_type"] = 1
    out["peak_id"] = -1
    out["dense_label"] = -1
    out["centroid_y"] = out["coord_y"]
    out["centroid_x"] = out["coord_x"]
    out["norm_y"] = norm_y
    out["norm_x"] = norm_x
    out["radial_distance_px"] = radial_px
    out["radial_distance_norm"] = radial_norm
    out["size_px"] = np.nan
    out["intensity_min"] = np.nan
    out["intensity_max"] = np.nan
    out["intensity_mean"] = np.nan
    out["intensity_median"] = np.nan
    out["intensity_std"] = np.nan
    out["signed_angle_deg"] = signed_angle_deg
    out["dist_prev_boundary"] = dist_prev
    out["dist_next_boundary"] = dist_next
    out["dist_to_nearest_peak"] = dist_to_peak
    keep_cols = [
        "frame",
        "graph_node_id",
        "node_type",
        "peak_id",
        "dense_label",
        "boundary_id",
        "boundary_order",
        "coord_y",
        "coord_x",
        "centroid_y",
        "centroid_x",
        "rel_y",
        "rel_x",
        "norm_y",
        "norm_x",
        "radial_distance_px",
        "radial_distance_norm",
        "size_px",
        "intensity_min",
        "intensity_max",
        "intensity_mean",
        "intensity_median",
        "intensity_std",
        "signed_angle_deg",
        "dist_prev_boundary",
        "dist_next_boundary",
        "dist_to_nearest_peak",
        "nucleus_centroid_y",
        "nucleus_centroid_x",
    ]
    return out[keep_cols].sort_values("boundary_order").reset_index(drop=True)
