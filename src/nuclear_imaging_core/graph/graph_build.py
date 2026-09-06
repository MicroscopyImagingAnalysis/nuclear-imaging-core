from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import networkx as nx
import numpy as np
import pandas as pd

from .dense_regions import DenseFrameResult
from .features.edge_features import (
    build_boundary_boundary_edges,
    build_peak_boundary_edges,
    build_peak_peak_edges,
)
from .features.graph_features import compute_graph_features
from .features.node_features import build_boundary_node_table, build_peak_node_table


@dataclass
class FrameGraphBundle:
    frame_name: str
    graph: nx.Graph
    node_table: pd.DataFrame
    edge_table: pd.DataFrame
    graph_attrs: dict


def build_graph_from_nodes(node_table: pd.DataFrame, frame_name: str) -> nx.Graph:
    """
    Build a graph directly from a unified node table.
    """
    G = nx.Graph()
    G.graph["frame"] = frame_name

    if node_table.empty:
        return G

    for r in node_table.itertuples(index=False):
        attrs = {k: getattr(r, k) for k in node_table.columns if k != "graph_node_id"}
        node_id = int(getattr(r, "graph_node_id"))
        G.add_node(node_id, **attrs)

    return G


def _compose_unified_node_table(peak_nodes: pd.DataFrame, boundary_nodes: pd.DataFrame) -> pd.DataFrame:
    tables = []
    if not peak_nodes.empty:
        tables.append(peak_nodes.copy())
    if not boundary_nodes.empty:
        tables.append(boundary_nodes.copy())
    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True, sort=False)


def _compose_edge_table(
    peak_nodes: pd.DataFrame,
    boundary_nodes: pd.DataFrame,
    peak_peak_thresh: float,
    peak_boundary_thresh: float,
) -> pd.DataFrame:
    parts = [
        build_peak_peak_edges(peak_nodes, distance_threshold_norm=peak_peak_thresh),
        build_peak_boundary_edges(
            peak_nodes,
            boundary_nodes,
            distance_threshold_norm=peak_boundary_thresh,
        ),
        build_boundary_boundary_edges(boundary_nodes),
    ]
    parts = [p for p in parts if p is not None and not p.empty]
    if not parts:
        return pd.DataFrame(
            columns=[
                "src_node_id",
                "dst_node_id",
                "edge_type",
                "distance_euclidean",
            ]
        )
    out = pd.concat(parts, ignore_index=True)
    out["src_node_id"] = out["src_node_id"].astype(int)
    out["dst_node_id"] = out["dst_node_id"].astype(int)
    out["edge_type"] = out["edge_type"].astype(int)
    out["distance_euclidean"] = out["distance_euclidean"].astype(float)
    return out


def _build_nx_graph(
    frame_name: str,
    node_table: pd.DataFrame,
    edge_table: pd.DataFrame,
    graph_attrs: dict,
) -> nx.Graph:
    G = nx.Graph()
    G.graph.update({"frame": frame_name, **graph_attrs})

    if not node_table.empty:
        for row in node_table.to_dict("records"):
            node_id = int(row["graph_node_id"])
            attrs = {k: v for k, v in row.items() if k != "graph_node_id"}
            G.add_node(node_id, **attrs)

    if not edge_table.empty:
        for row in edge_table.to_dict("records"):
            G.add_edge(
                int(row["src_node_id"]),
                int(row["dst_node_id"]),
                edge_type=int(row["edge_type"]),
                distance_euclidean=float(row["distance_euclidean"]),
            )

    return G


def build_frame_graph_bundle(
    frame_result: DenseFrameResult,
    frame_name: str,
    peak_peak_distance_threshold_norm: float = 0.25,
    peak_boundary_distance_threshold_norm: float = 0.25,
) -> FrameGraphBundle:
    peak_nodes = build_peak_node_table(frame_result)
    boundary_nodes = build_boundary_node_table(frame_result, peak_nodes=peak_nodes)
    node_table = _compose_unified_node_table(peak_nodes, boundary_nodes)
    edge_table = _compose_edge_table(
        peak_nodes=peak_nodes,
        boundary_nodes=boundary_nodes,
        peak_peak_thresh=float(peak_peak_distance_threshold_norm),
        peak_boundary_thresh=float(peak_boundary_distance_threshold_norm),
    )
    graph_attrs = compute_graph_features(
        node_table=node_table,
        edge_table=edge_table,
        frame_result=frame_result,
    )
    graph = _build_nx_graph(frame_name, node_table, edge_table, graph_attrs)
    return FrameGraphBundle(
        frame_name=frame_name,
        graph=graph,
        node_table=node_table,
        edge_table=edge_table,
        graph_attrs=graph_attrs,
    )


def graph_edge_table(G: nx.Graph) -> pd.DataFrame:
    rows = []
    for u, v, d in G.edges(data=True):
        rows.append(
            {
                "frame": G.graph.get("frame", ""),
                "src_node_id": int(u),
                "dst_node_id": int(v),
                "edge_type": int(d.get("edge_type", -1)),
                "distance_euclidean": float(d.get("distance_euclidean", np.nan)),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["frame", "src_node_id", "dst_node_id", "edge_type", "distance_euclidean"]
        )
    return pd.DataFrame(rows)


def build_graph_bundles_for_triplet(
    frame_results: Dict[str, DenseFrameResult],
    frame_order: tuple[str, str, str] = ("ref", "dec", "fin"),
    peak_peak_distance_threshold_norm: float = 0.25,
    peak_boundary_distance_threshold_norm: float = 0.25,
) -> Dict[str, FrameGraphBundle]:
    bundles: Dict[str, FrameGraphBundle] = {}
    for frame_name in frame_order:
        fr = frame_results[frame_name]
        bundles[frame_name] = build_frame_graph_bundle(
            frame_result=fr,
            frame_name=frame_name,
            peak_peak_distance_threshold_norm=peak_peak_distance_threshold_norm,
            peak_boundary_distance_threshold_norm=peak_boundary_distance_threshold_norm,
        )
    return bundles


def build_graphs_for_triplet(
    frame_results: Dict[str, DenseFrameResult],
    frame_order: tuple[str, str, str] = ("ref", "dec", "fin"),
    peak_peak_distance_threshold_norm: float = 0.25,
    peak_boundary_distance_threshold_norm: float = 0.25,
) -> Dict[str, nx.Graph]:
    bundles = build_graph_bundles_for_triplet(
        frame_results=frame_results,
        frame_order=frame_order,
        peak_peak_distance_threshold_norm=peak_peak_distance_threshold_norm,
        peak_boundary_distance_threshold_norm=peak_boundary_distance_threshold_norm,
    )
    return {k: v.graph for k, v in bundles.items()}
