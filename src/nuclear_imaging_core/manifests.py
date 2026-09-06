from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


DEFAULT_MANIFEST_STEMS = ("composed_manifest", "crop_manifest")


def utc_timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _manifest_path(output_dir: str | Path, stem: str) -> Path:
    return Path(output_dir) / f"{stem}.csv"


def _collect_previous_manifest_versions(output_dir: str | Path, stem: str) -> list[Path]:
    out_dir = Path(output_dir)
    versioned = sorted(out_dir.glob(f"{stem}_*.csv"))
    if versioned:
        return versioned
    previous_path = _manifest_path(out_dir, stem)
    return [previous_path] if previous_path.exists() else []


def _collect_experiment_manifest_paths(output_dir: str | Path, stem: str) -> list[Path]:
    out_dir = Path(output_dir)
    if not out_dir.exists():
        return []
    paths: list[Path] = []
    for child in sorted(out_dir.iterdir()):
        if not child.is_dir():
            continue
        manifest_path = _manifest_path(child, stem)
        if manifest_path.exists():
            paths.append(manifest_path)
    return paths


def _read_manifest_frame(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def compose_saved_manifests(
    output_dir: str | Path,
    *,
    stems: tuple[str, ...] = DEFAULT_MANIFEST_STEMS,
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}
    for stem in stems:
        source_paths = _collect_experiment_manifest_paths(out_dir, stem)
        if not source_paths:
            source_paths = _collect_previous_manifest_versions(out_dir, stem)
        if not source_paths:
            continue
        frames = [_read_manifest_frame(path) for path in source_paths if path.exists()]
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
        aggregate_path = _manifest_path(out_dir, stem)
        combined.to_csv(aggregate_path, index=False)
        saved[stem] = aggregate_path
    return saved


def write_versioned_manifests(
    output_dir: str | Path,
    manifest_tables: dict[str, pd.DataFrame],
    *,
    timestamp: str | None = None,
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for stem, df in manifest_tables.items():
        if df is None:
            continue
        path = _manifest_path(out_dir, stem)
        df.to_csv(path, index=False)
        written[stem] = path
    return written


__all__ = [
    "DEFAULT_MANIFEST_STEMS",
    "compose_saved_manifests",
    "utc_timestamp_slug",
    "write_versioned_manifests",
]
