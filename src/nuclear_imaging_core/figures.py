"""Reference/generated figure resolution for reproducible notebooks."""

from __future__ import annotations

from pathlib import Path


def resolve_figure(
    repository_root: str | Path,
    relative_path: str | Path,
    *,
    require_existing: bool = True,
) -> Path:
    """Prefer a generated figure and otherwise return its reference rendering."""
    root = Path(repository_root)
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError("relative_path must be repository-relative")
    generated = root / "figures" / "generated" / relative
    if generated.exists():
        return generated
    reference = root / "figures" / "reference" / relative
    if require_existing and not reference.exists():
        raise FileNotFoundError(
            f"No generated or reference figure for {relative}; checked {generated} and {reference}"
        )
    return reference


def generated_figure(
    repository_root: str | Path,
    relative_path: str | Path,
    *,
    create_parent: bool = True,
) -> Path:
    """Return the writable path parallel to a reference rendering."""
    root = Path(repository_root)
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError("relative_path must be repository-relative")
    destination = root / "figures" / "generated" / relative
    if create_parent:
        destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


__all__ = ["generated_figure", "resolve_figure"]
