from __future__ import annotations

from typing import Sequence


TRIPLET_FRAME_ALIASES = ("ref", "dec", "fin")
QUAD_FRAME_ALIASES = ("pre", "ref", "dec", "fin")


def default_frame_order(n_frames: int) -> tuple[str, ...]:
    n_frames = int(n_frames)
    if n_frames < 1:
        raise ValueError("n_frames must be >= 1")
    if n_frames == 3:
        return TRIPLET_FRAME_ALIASES
    if n_frames == 4:
        return QUAD_FRAME_ALIASES
    return tuple(str(idx) for idx in range(n_frames))


def canonicalize_frame_order(
    frame_order: Sequence[str | int] | None = None,
    *,
    n_frames: int | None = None,
) -> tuple[str, ...]:
    if frame_order is None:
        if n_frames is None:
            raise ValueError("Either frame_order or n_frames must be provided.")
        frame_order = default_frame_order(int(n_frames))
    canonical = tuple(str(name) for name in frame_order)
    if len(canonical) == 0:
        raise ValueError("frame_order must contain at least one frame.")
    if len(set(canonical)) != len(canonical):
        raise ValueError(f"frame_order contains duplicates: {canonical}")
    return canonical


def build_frame_alias_lookup(frame_order: Sequence[str | int]) -> dict[str, str]:
    canonical = canonicalize_frame_order(frame_order)
    lookup: dict[str, str] = {}
    for idx, name in enumerate(canonical):
        lookup[name] = name
        lookup[str(idx)] = name

    if len(canonical) == 3:
        aliases = TRIPLET_FRAME_ALIASES
    else:
        aliases = QUAD_FRAME_ALIASES[: min(len(canonical), len(QUAD_FRAME_ALIASES))]

    for alias, name in zip(aliases, canonical):
        lookup.setdefault(alias, name)

    # When callers omit an explicit reference, treat the first frame as the
    # anchor. For 4-point data this is the `pre` frame by convention.
    lookup.setdefault("pre", canonical[0])
    return lookup


def resolve_frame_name(frame: str | int, frame_order: Sequence[str | int]) -> str:
    lookup = build_frame_alias_lookup(frame_order)
    key = str(frame)
    if key not in lookup:
        raise ValueError(f"Unknown frame {frame!r}; valid names are {tuple(lookup)}")
    return lookup[key]


def resolve_reference_frame(
    frame_order: Sequence[str | int],
    reference_frame: str | int | None = None,
) -> str:
    canonical = canonicalize_frame_order(frame_order)
    if reference_frame is None:
        return build_frame_alias_lookup(canonical)["pre"]
    return resolve_frame_name(reference_frame, canonical)


def frame_pairs_from_reference(
    frame_order: Sequence[str | int],
    reference_frame: str | int | None = None,
    *,
    include_self: bool = True,
) -> tuple[tuple[str, str], ...]:
    canonical = canonicalize_frame_order(frame_order)
    ref = resolve_reference_frame(canonical, reference_frame=reference_frame)
    pairs: list[tuple[str, str]] = []
    for frame_name in canonical:
        if include_self or frame_name != ref:
            pairs.append((ref, frame_name))
    return tuple(pairs)


__all__ = [
    "TRIPLET_FRAME_ALIASES",
    "QUAD_FRAME_ALIASES",
    "build_frame_alias_lookup",
    "canonicalize_frame_order",
    "default_frame_order",
    "frame_pairs_from_reference",
    "resolve_frame_name",
    "resolve_reference_frame",
]
