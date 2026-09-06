from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterable

import pandas as pd

from ..metadata import ensure_meta_fields

if TYPE_CHECKING:
    from ..graph_build import FrameGraphBundle


def _triplet_id_from_meta(meta: dict) -> str:
    if "triplet_id" in meta and meta["triplet_id"] is not None:
        return str(meta["triplet_id"])
    return "|".join(
        [
            str(meta.get("experiment_id", "")),
            str(meta.get("nd2_prefix", "")),
            str(meta.get("crop_id", "")),
        ]
    )


def _meta_prefix(meta: dict, frame_name: str) -> dict:
    meta_full = ensure_meta_fields(meta)
    out = {
        "experiment_id": meta_full.get("experiment_id"),
        "cell": meta_full.get("cell"),
        "condition": meta_full.get("condition"),
        "batch": meta_full.get("batch"),
        "nd2_prefix": meta_full.get("nd2_prefix"),
        "crop_id": meta_full.get("crop_id"),
        "crop_path": meta_full.get("crop_path"),
        "triplet_id": _triplet_id_from_meta(meta_full),
        "frame": frame_name,
    }
    return out


def frame_graph_feature_row(meta: dict, bundle: FrameGraphBundle) -> dict:
    row = _meta_prefix(meta, bundle.frame_name)
    row.update(bundle.graph_attrs)
    return row


def aggregate_frame_graph_feature_rows(meta: dict, bundles: Dict[str, FrameGraphBundle]) -> pd.DataFrame:
    rows = [frame_graph_feature_row(meta, bundle) for bundle in bundles.values()]
    return pd.DataFrame(rows)


def aggregate_graph_bundle_tables(meta: dict, bundles: Dict[str, FrameGraphBundle]) -> dict:
    node_parts = []
    edge_parts = []
    graph_parts = []

    for frame_name, bundle in bundles.items():
        meta_cols = _meta_prefix(meta, frame_name)

        ndf = bundle.node_table.copy()
        if not ndf.empty:
            for col, val in reversed(list(meta_cols.items())):
                if col in ndf.columns:
                    ndf[col] = val
                else:
                    ndf.insert(0, col, val)
            node_parts.append(ndf)

        edf = bundle.edge_table.copy()
        if not edf.empty:
            for col, val in reversed(list(meta_cols.items())):
                if col in edf.columns:
                    edf[col] = val
                else:
                    edf.insert(0, col, val)
            edge_parts.append(edf)

        graph_parts.append(pd.DataFrame([frame_graph_feature_row(meta, bundle)]))

    return {
        "graph_nodes": pd.concat(node_parts, ignore_index=True) if node_parts else pd.DataFrame(),
        "graph_edges": pd.concat(edge_parts, ignore_index=True) if edge_parts else pd.DataFrame(),
        "frame_graph_features": pd.concat(graph_parts, ignore_index=True) if graph_parts else pd.DataFrame(),
    }


def aggregate_many_feature_tables(per_triplet_tables: Iterable[dict]) -> dict:
    keys = ("graph_nodes", "graph_edges", "frame_graph_features")
    out = {}
    for key in keys:
        parts = [tbl[key] for tbl in per_triplet_tables if key in tbl and tbl[key] is not None and not tbl[key].empty]
        out[key] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return out
