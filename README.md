# nuclear-imaging-core

Reusable building blocks for quantitative analysis of 2D and 3D microscopy
images. The package covers image loading, segmentation, object measurements,
radial and spatial analysis, graph construction, tracking, manifests and quality
control.

## Install

```bash
python -m pip install .
```

StarDist and RGB/AP workflows use optional dependency groups:

```bash
python -m pip install ".[stardist,rgb]"
```

## Segmentation

```python
from nuclear_imaging_core.segmentation import SegmentationConfig, segment_frame

config = SegmentationConfig(
    analysis_mode="3D",
    segmentation_method="multiotsu",
    min_size=100,
)
labels, mask = segment_frame(image, config)
```

## Measurements and representations

- `features.two_d` and `features.three_d` expose morphology, intensity, texture,
  curvature and radial measurements.
- `measurements` provides concise object tables and normalized radial profiles.
- `graph` identifies dense chromatin regions, builds spatial graphs, tracks
  nodes across timepoints and aggregates graph descriptors.
- `manifests` and `qc` support repeatable batch processing.

Geometric thresholds and radial distances are expressed in pixels, keeping all
analysis choices explicit and easy to adapt to a particular acquisition.
