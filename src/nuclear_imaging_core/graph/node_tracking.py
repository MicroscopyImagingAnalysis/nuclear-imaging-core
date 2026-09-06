from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class TrackingConfig:
    w_position: float = 1.0
    w_size: float = 1.0
    # This diagnostic threshold does not apply hard gating.
    candidate_radius_px: float = 20.0


def _normalized_match_weights(cfg: TrackingConfig) -> tuple[float, float]:
    """Normalize position/size weights to sum to 1.0."""
    wp = max(float(cfg.w_position), 0.0)
    ws = max(float(cfg.w_size), 0.0)
    total = wp + ws
    if total <= 0.0:
        return 0.5, 0.5
    return wp / total, ws / total


def _empty_matches_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "src_frame",
            "dst_frame",
            "src_node_id",
            "dst_node_id",
            "dst_node_id_original",
            "cost",
            "distance_px",
            "pos_diff_norm",
            "size_diff_norm",
            "primary_daughter",
        ]
    )


def _nucleus_centroid(nodes: pd.DataFrame) -> tuple[float, float]:
    if nodes.empty:
        return 0.0, 0.0
    if {"nucleus_centroid_y", "nucleus_centroid_x"}.issubset(nodes.columns):
        return float(nodes["nucleus_centroid_y"].iloc[0]), float(nodes["nucleus_centroid_x"].iloc[0])
    return float(nodes["centroid_y"].mean()), float(nodes["centroid_x"].mean())


def _radius_bounds(mask: np.ndarray | None, cy: float, cx: float, nodes: pd.DataFrame) -> tuple[float, float]:
    eps = 1e-12
    if mask is not None and np.count_nonzero(mask) > 0:
        ys, xs = np.nonzero(mask)
        r_mask = np.hypot(ys.astype(np.float64) - cy, xs.astype(np.float64) - cx)
        if r_mask.size > 0:
            rmin = float(np.min(r_mask))
            rmax = float(np.max(r_mask))
            if (rmax - rmin) > eps:
                return rmin, rmax

    if nodes.empty:
        return 0.0, 1.0

    dy = nodes["centroid_y"].to_numpy(dtype=np.float64) - cy
    dx = nodes["centroid_x"].to_numpy(dtype=np.float64) - cx
    r_nodes = np.hypot(dy, dx)
    rmin = float(np.min(r_nodes))
    rmax = float(np.max(r_nodes))
    if (rmax - rmin) <= eps:
        return 0.0, 1.0
    return rmin, rmax


def _size_norm_per_frame(nodes: pd.DataFrame) -> np.ndarray:
    sizes = nodes["size_px"].to_numpy(dtype=np.float64)
    if sizes.size == 0:
        return np.zeros(0, dtype=np.float64)
    smin = float(np.min(sizes))
    smax = float(np.max(sizes))
    if (smax - smin) <= 1e-12:
        return np.full_like(sizes, 0.5, dtype=np.float64)
    return np.clip((sizes - smin) / (smax - smin), 0.0, 1.0)


def _frame_matching_features(nodes: pd.DataFrame, mask: np.ndarray | None = None) -> pd.DataFrame:
    """
    Build frame-local normalized matching features:
    - position relative to nucleus centroid, scaled to [0,1] using radial min/max
    - size normalized to [0,1] per frame
    """
    if nodes.empty:
        return pd.DataFrame(columns=["node_id", "x_norm", "y_norm", "size_norm"])

    cy, cx = _nucleus_centroid(nodes)
    rmin, rmax = _radius_bounds(mask, cy, cx, nodes)
    denom_r = max(rmax - rmin, 1e-12)

    dy = nodes["centroid_y"].to_numpy(dtype=np.float64) - cy
    dx = nodes["centroid_x"].to_numpy(dtype=np.float64) - cx
    r_node = np.hypot(dy, dx)
    r_scaled = np.clip((r_node - rmin) / denom_r, 0.0, 1.0)

    ux = np.divide(dx, r_node, out=np.zeros_like(dx), where=r_node > 1e-12)
    uy = np.divide(dy, r_node, out=np.zeros_like(dy), where=r_node > 1e-12)

    x_signed = ux * r_scaled
    y_signed = uy * r_scaled

    x_norm = np.clip((x_signed + 1.0) / 2.0, 0.0, 1.0)
    y_norm = np.clip((y_signed + 1.0) / 2.0, 0.0, 1.0)
    size_norm = _size_norm_per_frame(nodes)

    return pd.DataFrame(
        {
            "node_id": nodes["node_id"].astype(int).to_numpy(),
            "x_norm": x_norm,
            "y_norm": y_norm,
            "size_norm": size_norm,
        }
    )


def _match_ref_to_target(
    ref_nodes: pd.DataFrame,
    target_nodes: pd.DataFrame,
    src_frame: str,
    dst_frame: str,
    cfg: TrackingConfig,
    ref_mask: np.ndarray | None = None,
    target_mask: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if ref_nodes.empty or target_nodes.empty:
        out_target = target_nodes.copy()
        if not out_target.empty:
            out_target["node_id_original"] = out_target["node_id"].astype(int)
            out_target["node_id"] = out_target["node_id"].astype(int)
        return _empty_matches_df(), out_target

    ref_feat = _frame_matching_features(ref_nodes, mask=ref_mask)
    target_feat = _frame_matching_features(target_nodes, mask=target_mask)

    src_pos = ref_feat[["x_norm", "y_norm"]].to_numpy(dtype=np.float64)
    dst_pos = target_feat[["x_norm", "y_norm"]].to_numpy(dtype=np.float64)
    src_size = ref_feat["size_norm"].to_numpy(dtype=np.float64)
    dst_size = target_feat["size_norm"].to_numpy(dtype=np.float64)

    pos_diff = np.linalg.norm(src_pos[:, None, :] - dst_pos[None, :, :], axis=2) / np.sqrt(2.0)
    size_diff = np.abs(src_size[:, None] - dst_size[None, :])
    wpos, wsize = _normalized_match_weights(cfg)
    cost = (wpos * pos_diff) + (wsize * size_diff)

    row_ind, col_ind = linear_sum_assignment(cost)

    dst_to_ref: dict[int, int] = {}
    rows = []
    for ri, ci in zip(row_ind, col_ind):
        ref_id = int(ref_feat.iloc[int(ri)]["node_id"])
        dst_orig = int(target_feat.iloc[int(ci)]["node_id"])
        dst_to_ref[dst_orig] = ref_id
        rows.append(
            {
                "src_frame": src_frame,
                "dst_frame": dst_frame,
                "src_node_id": ref_id,
                "dst_node_id": ref_id,
                "dst_node_id_original": dst_orig,
                "cost": float(cost[ri, ci]),
                "distance_px": float(pos_diff[ri, ci]),
                "pos_diff_norm": float(pos_diff[ri, ci]),
                "size_diff_norm": float(size_diff[ri, ci]),
                "primary_daughter": True,
            }
        )

    out_target = target_nodes.copy()
    out_target["node_id_original"] = out_target["node_id"].astype(int)
    max_ref_id = int(ref_nodes["node_id"].max()) if not ref_nodes.empty else 0
    next_new = max_ref_id + 1

    reassigned_ids = []
    for original_id in out_target["node_id_original"].tolist():
        oid = int(original_id)
        if oid in dst_to_ref:
            reassigned_ids.append(int(dst_to_ref[oid]))
        else:
            reassigned_ids.append(int(next_new))
            next_new += 1
    out_target["node_id"] = np.asarray(reassigned_ids, dtype=int)

    if not rows:
        return _empty_matches_df(), out_target

    matches = pd.DataFrame(rows).sort_values(["src_node_id", "dst_node_id_original"]).reset_index(drop=True)
    return matches, out_target


def match_nodes_between_frames(
    src_nodes: pd.DataFrame,
    dst_nodes: pd.DataFrame,
    src_frame: str,
    dst_frame: str,
    cfg: TrackingConfig,
    src_mask: np.ndarray | None = None,
    dst_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    matches, _ = _match_ref_to_target(
        ref_nodes=src_nodes,
        target_nodes=dst_nodes,
        src_frame=src_frame,
        dst_frame=dst_frame,
        cfg=cfg,
        ref_mask=src_mask,
        target_mask=dst_mask,
    )
    return matches


def track_triplet(
    frame_nodes: Dict[str, pd.DataFrame],
    cfg: TrackingConfig,
    frame_order: tuple[str, str, str] = ("ref", "dec", "fin"),
    frame_masks: Dict[str, np.ndarray] | None = None,
) -> dict:
    """
    Track nodes with ref-anchored IDs.

    Returns dict with:
    - matches_ref_dec
    - matches_dec_fin
    - tracked_nodes (dict of frame->nodes with node_id rewritten in dec/fin to align with ref)
    """
    ref, dec, fin = frame_order
    tracked = track_ordered_nodes(
        frame_nodes=frame_nodes,
        cfg=cfg,
        frame_order=frame_order,
        frame_masks=frame_masks,
    )
    return {
        "matches_ref_dec": tracked["matches_by_frame"].get(dec, _empty_matches_df()),
        "matches_dec_fin": tracked["matches_by_frame"].get(fin, _empty_matches_df()),
        "matches_ref_fin": tracked["matches_by_frame"].get(fin, _empty_matches_df()),
        "tracked_nodes": tracked["tracked_nodes"],
    }


def track_ordered_nodes(
    frame_nodes: Dict[str, pd.DataFrame],
    cfg: TrackingConfig,
    frame_order: Sequence[str],
    frame_masks: Dict[str, np.ndarray] | None = None,
) -> dict:
    frame_order = tuple(str(name) for name in frame_order)
    if len(frame_order) < 2:
        raise ValueError("frame_order must contain at least two frames.")

    masks = frame_masks or {}
    anchor = frame_order[0]
    anchor_nodes = frame_nodes.get(anchor, pd.DataFrame()).copy()

    tracked_nodes: dict[str, pd.DataFrame] = {anchor: anchor_nodes.copy()}
    matches_by_frame: dict[str, pd.DataFrame] = {}

    for frame_name in frame_order[1:]:
        target_nodes = frame_nodes.get(frame_name, pd.DataFrame()).copy()
        matches, reassigned_nodes = _match_ref_to_target(
            ref_nodes=anchor_nodes,
            target_nodes=target_nodes,
            src_frame=anchor,
            dst_frame=frame_name,
            cfg=cfg,
            ref_mask=masks.get(anchor),
            target_mask=masks.get(frame_name),
        )
        tracked_nodes[frame_name] = reassigned_nodes.copy()
        matches_by_frame[frame_name] = matches

    for frame_name, nodes in tracked_nodes.items():
        if not nodes.empty:
            tracked_nodes[frame_name]["track_id"] = nodes["node_id"].astype(int)

    return {
        "anchor_frame": anchor,
        "frame_order": frame_order,
        "matches_by_frame": matches_by_frame,
        "tracked_nodes": tracked_nodes,
    }
