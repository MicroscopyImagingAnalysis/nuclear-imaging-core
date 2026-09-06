from pathlib import Path

import numpy as np

from nuclear_imaging_core.figures import generated_figure, resolve_figure
from nuclear_imaging_core.segmentation import normalize_image, segment3d_multiotsu_simple
from nuclear_imaging_core.timepoints import canonicalize_frame_order, default_frame_order


def test_timepoint_defaults_follow_ordered_frames():
    assert default_frame_order(3) == ("ref", "dec", "fin")
    assert default_frame_order(4) == ("pre", "ref", "dec", "fin")
    assert canonicalize_frame_order((0, 1, 2)) == ("0", "1", "2")


def test_normalization_and_simple_3d_segmentation_shapes():
    image = np.zeros((3, 12, 12), dtype=np.float32)
    image[:, 2:10, 2:10] = 0.5
    image[:, 4:8, 4:8] = 1.0
    normalized = normalize_image(image)
    labels, mask = segment3d_multiotsu_simple(image, min_size=2)
    assert normalized.min() == 0.0
    assert normalized.max() == 1.0
    assert labels.shape == image.shape
    assert mask.shape == image.shape
    assert labels.dtype == np.uint16


def test_figure_resolution_prefers_generated_and_falls_back(tmp_path: Path):
    relative = Path("Fig4/Fig4a.png")
    reference = tmp_path / "figures/reference" / relative
    generated = tmp_path / "figures/generated" / relative
    reference.parent.mkdir(parents=True)
    reference.write_bytes(b"reference")
    assert resolve_figure(tmp_path, relative) == reference
    generated.parent.mkdir(parents=True)
    generated.write_bytes(b"generated")
    assert resolve_figure(tmp_path, relative) == generated
    assert generated_figure(tmp_path, relative) == generated
