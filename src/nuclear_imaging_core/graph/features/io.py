from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from .aggregate_features import _triplet_id_from_meta, frame_graph_feature_row

if TYPE_CHECKING:
    from ..graph_build import FrameGraphBundle


def _sanitize_name(value: str) -> str:
    return str(value).replace("/", "_").replace(" ", "_")


def save_frame_graph_bundle(bundle: FrameGraphBundle, out_dir: str | Path, meta: dict) -> Path:
    triplet_id = _sanitize_name(_triplet_id_from_meta(meta))
    frame_name = _sanitize_name(bundle.frame_name)
    bundle_dir = Path(out_dir) / triplet_id / frame_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    bundle.node_table.to_csv(bundle_dir / "nodes.csv", index=False)
    bundle.edge_table.to_csv(bundle_dir / "edges.csv", index=False)

    with (bundle_dir / "graph_attrs.json").open("w", encoding="utf-8") as fh:
        json.dump(bundle.graph_attrs, fh, indent=2)

    with (bundle_dir / "metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(frame_graph_feature_row(meta, bundle), fh, indent=2)

    return bundle_dir


def save_triplet_graph_bundles(bundles: dict[str, FrameGraphBundle], out_dir: str | Path, meta: dict) -> dict[str, Path]:
    saved = {}
    for frame_name, bundle in bundles.items():
        saved[frame_name] = save_frame_graph_bundle(bundle, out_dir=out_dir, meta=meta)
    return saved
