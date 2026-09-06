from __future__ import annotations

from typing import Dict, List

import networkx as nx
import numpy as np
import pandas as pd

from .metadata import ensure_meta_fields


def summarize_nodes(frame: str, node_df: pd.DataFrame) -> dict:
    if node_df.empty:
        return {
            "frame": frame,
            "n_nodes": 0,
            "size_mean_px": np.nan,
            "size_std_px": np.nan,
            "size_sum_px": 0,
        }
    return {
        "frame": frame,
        "n_nodes": int(len(node_df)),
        "size_mean_px": float(node_df["size_px"].mean()),
        "size_std_px": float(node_df["size_px"].std(ddof=0)),
        "size_sum_px": int(node_df["size_px"].sum()),
    }


def summarize_graph(frame: str, G: nx.Graph) -> dict:
    if G.number_of_nodes() == 0:
        return {
            "frame": frame,
            "n_nodes": 0,
            "n_edges": 0,
            "mean_pair_distance_px": np.nan,
            "median_pair_distance_px": np.nan,
        }

    dists = [float(d.get("distance_px", np.nan)) for _, _, d in G.edges(data=True)]
    if len(dists) == 0:
        mean_d = np.nan
        median_d = np.nan
    else:
        mean_d = float(np.nanmean(dists))
        median_d = float(np.nanmedian(dists))

    return {
        "frame": frame,
        "n_nodes": int(G.number_of_nodes()),
        "n_edges": int(G.number_of_edges()),
        "mean_pair_distance_px": mean_d,
        "median_pair_distance_px": median_d,
    }


def aggregate_triplet_outputs(
    meta: dict,
    frame_results: Dict[str, object],
    graphs: Dict[str, nx.Graph],
    tracking_output: dict,
    qc_row: dict | None = None,
) -> dict:
    """
    Create normalized tables for one crop triplet.

    Returns dict of DataFrames:
    - nodes
    - frame_node_summary
    - frame_graph_summary
    - matches_ref_dec
    - matches_dec_fin
    - triplet_summary
    """
    meta_full = ensure_meta_fields(meta)
    exp_id = str(meta_full.get("experiment_id"))
    cell = str(meta_full.get("cell"))
    condition = str(meta_full.get("condition"))
    batch = str(meta_full.get("batch"))
    nd2_prefix = str(meta_full.get("nd2_prefix"))
    crop_id = int(meta_full.get("crop_id"))
    crop_path = str(meta_full.get("crop_path"))

    node_tables: List[pd.DataFrame] = []
    node_summary_rows: List[dict] = []
    graph_summary_rows: List[dict] = []

    for frame_name, fr in frame_results.items():
        ndf = fr.node_table.copy()
        ndf.insert(0, "experiment_id", exp_id)
        ndf.insert(1, "cell", cell)
        ndf.insert(2, "condition", condition)
        ndf.insert(3, "batch", batch)
        ndf.insert(4, "nd2_prefix", nd2_prefix)
        ndf.insert(5, "crop_id", crop_id)
        ndf.insert(6, "crop_path", crop_path)
        ndf.insert(7, "threshold", float(fr.threshold))
        node_tables.append(ndf)

        node_summary = summarize_nodes(frame_name, fr.node_table)
        node_summary.update(
            {
                "experiment_id": exp_id,
                "cell": cell,
                "condition": condition,
                "batch": batch,
                "nd2_prefix": nd2_prefix,
                "crop_id": crop_id,
            }
        )
        node_summary_rows.append(node_summary)

        graph_summary = summarize_graph(frame_name, graphs.get(frame_name, nx.Graph()))
        graph_summary.update(
            {
                "experiment_id": exp_id,
                "cell": cell,
                "condition": condition,
                "batch": batch,
                "nd2_prefix": nd2_prefix,
                "crop_id": crop_id,
            }
        )
        graph_summary_rows.append(graph_summary)

    nodes_df = pd.concat(node_tables, ignore_index=True) if node_tables else pd.DataFrame()
    frame_node_summary_df = pd.DataFrame(node_summary_rows)
    frame_graph_summary_df = pd.DataFrame(graph_summary_rows)

    m_rd = tracking_output.get("matches_ref_dec", pd.DataFrame()).copy()
    m_df = tracking_output.get("matches_dec_fin", pd.DataFrame()).copy()

    for m in (m_rd, m_df):
        if not m.empty:
            m.insert(0, "experiment_id", exp_id)
            m.insert(1, "cell", cell)
            m.insert(2, "condition", condition)
            m.insert(3, "batch", batch)
            m.insert(4, "nd2_prefix", nd2_prefix)
            m.insert(5, "crop_id", crop_id)

    triplet_row = {
        "experiment_id": exp_id,
        "cell": cell,
        "condition": condition,
        "batch": batch,
        "nd2_prefix": nd2_prefix,
        "crop_id": crop_id,
        "crop_path": crop_path,
        "n_matches_ref_dec": int(len(m_rd)),
        "n_matches_dec_fin": int(len(m_df)),
    }
    if qc_row is not None:
        for k, v in qc_row.items():
            if k in ("experiment_id", "nd2_prefix", "crop_id", "crop_path"):
                continue
            triplet_row[f"qc_{k}"] = v

    triplet_summary_df = pd.DataFrame([triplet_row])

    return {
        "nodes": nodes_df,
        "frame_node_summary": frame_node_summary_df,
        "frame_graph_summary": frame_graph_summary_df,
        "matches_ref_dec": m_rd,
        "matches_dec_fin": m_df,
        "triplet_summary": triplet_summary_df,
    }


def aggregate_experiment_outputs(per_triplet_tables: List[dict]) -> dict:
    keys = [
        "nodes",
        "frame_node_summary",
        "frame_graph_summary",
        "matches_ref_dec",
        "matches_dec_fin",
        "triplet_summary",
    ]

    out = {}
    for k in keys:
        parts = [x[k] for x in per_triplet_tables if k in x and x[k] is not None and not x[k].empty]
        out[k] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return out
