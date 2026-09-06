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

Measure the segmented objects and their centre-to-boundary intensity profile:

```python
from nuclear_imaging_core.measurements import normalized_radial_profile, region_feature_table

object_table = region_feature_table(image, labels)
radial_table = normalized_radial_profile(image, labels > 0, bins=10)
```

Convert dense intranuclear regions into a spatial graph:

```python
from nuclear_imaging_core.graph.dense_regions import DenseRegionConfig, segment_dense_regions_frame
from nuclear_imaging_core.graph.graph_build import build_frame_graph_bundle

dense = segment_dense_regions_frame(
    nuclear_crop,
    "reference",
    DenseRegionConfig(with_peaks=True, with_boundary_nodes=True),
)
graph_bundle = build_frame_graph_bundle(dense, "reference")
graph_features = graph_bundle.graph_attrs
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
