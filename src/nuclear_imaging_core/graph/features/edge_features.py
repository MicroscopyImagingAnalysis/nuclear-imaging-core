from __future__ import annotations

import numpy as np
import pandas as pd


def _pairwise_norm_dist(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    return np.linalg.norm(src[:, None, :] - dst[None, :, :], axis=2)


def build_peak_peak_edges(
    peak_nodes: pd.DataFrame,
    distance_threshold_norm: float = 0.25,
) -> pd.DataFrame:
    if peak_nodes is None or peak_nodes.empty or len(peak_nodes) < 2:
        return pd.DataFrame(columns=["src_node_id", "dst_node_id", "edge_type", "distance_euclidean"])

    coords = peak_nodes[["norm_y", "norm_x"]].to_numpy(dtype=np.float64)
    ids = peak_nodes["graph_node_id"].to_numpy(dtype=int)
    dist = _pairwise_norm_dist(coords, coords)
    rows = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            dij = float(dist[i, j])
            if dij <= float(distance_threshold_norm):
                rows.append(
                    {
                        "src_node_id": int(ids[i]),
                        "dst_node_id": int(ids[j]),
                        "edge_type": 0,
                        "distance_euclidean": dij,
                    }
                )
    return pd.DataFrame(rows)


def build_peak_boundary_edges(
    peak_nodes: pd.DataFrame,
    boundary_nodes: pd.DataFrame,
    distance_threshold_norm: float = 0.25,
) -> pd.DataFrame:
    if peak_nodes is None or peak_nodes.empty or boundary_nodes is None or boundary_nodes.empty:
        return pd.DataFrame(columns=["src_node_id", "dst_node_id", "edge_type", "distance_euclidean"])

    peak_coords = peak_nodes[["norm_y", "norm_x"]].to_numpy(dtype=np.float64)
    peak_ids = peak_nodes["graph_node_id"].to_numpy(dtype=int)
    boundary_coords = boundary_nodes[["norm_y", "norm_x"]].to_numpy(dtype=np.float64)
    boundary_ids = boundary_nodes["graph_node_id"].to_numpy(dtype=int)
    dist = _pairwise_norm_dist(peak_coords, boundary_coords)

    rows = []
    for i in range(len(peak_ids)):
        for j in range(len(boundary_ids)):
            dij = float(dist[i, j])
            if dij <= float(distance_threshold_norm):
                rows.append(
                    {
                        "src_node_id": int(peak_ids[i]),
                        "dst_node_id": int(boundary_ids[j]),
                        "edge_type": 1,
                        "distance_euclidean": dij,
                    }
                )
    return pd.DataFrame(rows)


def build_boundary_boundary_edges(boundary_nodes: pd.DataFrame) -> pd.DataFrame:
    if boundary_nodes is None or boundary_nodes.empty or len(boundary_nodes) < 2:
        return pd.DataFrame(columns=["src_node_id", "dst_node_id", "edge_type", "distance_euclidean"])

    ordered = boundary_nodes.sort_values("boundary_order").reset_index(drop=True)
    coords = ordered[["norm_y", "norm_x"]].to_numpy(dtype=np.float64)
    ids = ordered["graph_node_id"].to_numpy(dtype=int)
    n = len(ids)

    rows = []
    for i in range(n):
        j = (i + 1) % n
        if ids[i] == ids[j]:
            continue
        src = int(min(ids[i], ids[j]))
        dst = int(max(ids[i], ids[j]))
        dij = float(np.linalg.norm(coords[i] - coords[j]))
        rows.append(
            {
                "src_node_id": src,
                "dst_node_id": dst,
                "edge_type": 2,
                "distance_euclidean": dij,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["src_node_id", "dst_node_id", "edge_type", "distance_euclidean"])
    out = pd.DataFrame(rows).drop_duplicates(subset=["src_node_id", "dst_node_id", "edge_type"])
    return out.sort_values(["src_node_id", "dst_node_id"]).reset_index(drop=True)
