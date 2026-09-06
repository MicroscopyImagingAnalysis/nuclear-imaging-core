from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..dense_regions import DenseFrameResult


def _safe_stats(values: np.ndarray, prefix: str) -> dict:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
        }
    return {
        f"{prefix}_mean": float(np.mean(vals)),
        f"{prefix}_std": float(np.std(vals, ddof=0)),
        f"{prefix}_min": float(np.min(vals)),
        f"{prefix}_max": float(np.max(vals)),
    }


def _principal_axis(coords: np.ndarray, weights: np.ndarray | None = None) -> dict:
    pts = np.asarray(coords, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return {
            "angle_rad": np.nan,
            "major_var": np.nan,
            "minor_var": np.nan,
            "anisotropy": np.nan,
        }
    if weights is None:
        w = np.ones(pts.shape[0], dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if np.sum(w) <= 0:
            w = np.ones(pts.shape[0], dtype=np.float64)
    center = np.average(pts, axis=0, weights=w)
    centered = pts - center
    cov = np.cov(centered.T, aweights=w, ddof=0)
    if cov.shape != (2, 2):
        return {
            "angle_rad": np.nan,
            "major_var": np.nan,
            "minor_var": np.nan,
            "anisotropy": np.nan,
        }
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    major = float(max(eigvals[0], 0.0))
    minor = float(max(eigvals[1], 0.0))
    vec = eigvecs[:, 0]
    angle = float(math.atan2(vec[0], vec[1]))
    anisotropy = float(major / minor) if minor > 1e-12 else np.nan
    return {
        "angle_rad": angle,
        "major_var": major,
        "minor_var": minor,
        "anisotropy": anisotropy,
    }


def _axis_delta_abs(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return np.nan
    delta = abs(a - b)
    while delta > math.pi:
        delta -= math.pi
    return float(min(delta, math.pi - delta))


def compute_graph_features(
    node_table: pd.DataFrame,
    edge_table: pd.DataFrame,
    frame_result: DenseFrameResult,
) -> dict:
    peak_nodes = node_table[node_table["node_type"] == 0].copy() if not node_table.empty else pd.DataFrame()
    boundary_nodes = node_table[node_table["node_type"] == 1].copy() if not node_table.empty else pd.DataFrame()

    feat = {
        "graph_num_nodes": int(len(node_table)),
        "graph_num_edges": int(len(edge_table)),
        "num_peak_nodes": int(len(peak_nodes)),
        "num_boundary_nodes": int(len(boundary_nodes)),
        "num_edges_peak_peak": int((edge_table["edge_type"] == 0).sum()) if not edge_table.empty else 0,
        "num_edges_peak_boundary": int((edge_table["edge_type"] == 1).sum()) if not edge_table.empty else 0,
        "num_edges_boundary_boundary": int((edge_table["edge_type"] == 2).sum()) if not edge_table.empty else 0,
        "threshold": float(frame_result.threshold),
        "nucleus_centroid_y": float(frame_result.nucleus_centroid_y),
        "nucleus_centroid_x": float(frame_result.nucleus_centroid_x),
        "nucleus_area_px": int(np.count_nonzero(frame_result.nucleus_mask)),
        "dense_area_px": int(np.count_nonzero(frame_result.dense_binary)),
    }

    if not peak_nodes.empty:
        feat.update(_safe_stats(peak_nodes["size_px"].to_numpy(dtype=np.float64), "peak_size_px"))
        feat.update(_safe_stats(peak_nodes["radial_distance_norm"].to_numpy(dtype=np.float64), "peak_radial_norm"))
        feat.update(_safe_stats(peak_nodes["intensity_mean"].to_numpy(dtype=np.float64), "peak_intensity_mean"))
        feat.update(_safe_stats(peak_nodes["intensity_std"].to_numpy(dtype=np.float64), "peak_intensity_std"))
        feat["peak_centroid_norm_y_mean"] = float(np.mean(peak_nodes["norm_y"]))
        feat["peak_centroid_norm_x_mean"] = float(np.mean(peak_nodes["norm_x"]))
        peak_axis = _principal_axis(
            peak_nodes[["norm_y", "norm_x"]].to_numpy(dtype=np.float64),
            weights=peak_nodes["size_px"].to_numpy(dtype=np.float64),
        )
    else:
        feat.update(_safe_stats(np.asarray([], dtype=np.float64), "peak_size_px"))
        feat.update(_safe_stats(np.asarray([], dtype=np.float64), "peak_radial_norm"))
        feat.update(_safe_stats(np.asarray([], dtype=np.float64), "peak_intensity_mean"))
        feat.update(_safe_stats(np.asarray([], dtype=np.float64), "peak_intensity_std"))
        feat["peak_centroid_norm_y_mean"] = np.nan
        feat["peak_centroid_norm_x_mean"] = np.nan
        peak_axis = _principal_axis(np.zeros((0, 2), dtype=np.float64))

    if not boundary_nodes.empty:
        feat.update(_safe_stats(boundary_nodes["radial_distance_norm"].to_numpy(dtype=np.float64), "boundary_radial_norm"))
        feat.update(_safe_stats(boundary_nodes["signed_angle_deg"].to_numpy(dtype=np.float64), "boundary_signed_angle_deg"))
        feat.update(_safe_stats(boundary_nodes["dist_to_nearest_peak"].to_numpy(dtype=np.float64), "boundary_to_peak_dist"))
        feat.update(_safe_stats(boundary_nodes["dist_prev_boundary"].to_numpy(dtype=np.float64), "boundary_edge_len"))
        boundary_axis = _principal_axis(boundary_nodes[["norm_y", "norm_x"]].to_numpy(dtype=np.float64))
    else:
        feat.update(_safe_stats(np.asarray([], dtype=np.float64), "boundary_radial_norm"))
        feat.update(_safe_stats(np.asarray([], dtype=np.float64), "boundary_signed_angle_deg"))
        feat.update(_safe_stats(np.asarray([], dtype=np.float64), "boundary_to_peak_dist"))
        feat.update(_safe_stats(np.asarray([], dtype=np.float64), "boundary_edge_len"))
        boundary_axis = _principal_axis(np.zeros((0, 2), dtype=np.float64))

    feat["peak_axis_angle_rad"] = peak_axis["angle_rad"]
    feat["peak_axis_major_var"] = peak_axis["major_var"]
    feat["peak_axis_minor_var"] = peak_axis["minor_var"]
    feat["peak_axis_anisotropy"] = peak_axis["anisotropy"]
    feat["boundary_axis_angle_rad"] = boundary_axis["angle_rad"]
    feat["boundary_axis_major_var"] = boundary_axis["major_var"]
    feat["boundary_axis_minor_var"] = boundary_axis["minor_var"]
    feat["boundary_axis_anisotropy"] = boundary_axis["anisotropy"]
    feat["peak_boundary_axis_delta_abs_rad"] = _axis_delta_abs(
        peak_axis["angle_rad"],
        boundary_axis["angle_rad"],
    )

    if not edge_table.empty:
        feat.update(_safe_stats(edge_table["distance_euclidean"].to_numpy(dtype=np.float64), "edge_distance"))
        for edge_type, label in ((0, "pp"), (1, "pb"), (2, "bb")):
            vals = edge_table.loc[edge_table["edge_type"] == edge_type, "distance_euclidean"].to_numpy(dtype=np.float64)
            feat.update(_safe_stats(vals, f"edge_distance_{label}"))
    else:
        feat.update(_safe_stats(np.asarray([], dtype=np.float64), "edge_distance"))
        feat.update(_safe_stats(np.asarray([], dtype=np.float64), "edge_distance_pp"))
        feat.update(_safe_stats(np.asarray([], dtype=np.float64), "edge_distance_pb"))
        feat.update(_safe_stats(np.asarray([], dtype=np.float64), "edge_distance_bb"))

    return feat
