import numpy as np

from nuclear_imaging_core.measurements import describe_image, normalized_radial_profile, region_feature_table


def test_image_description_and_region_table():
    image = np.arange(36, dtype=np.float32).reshape(6, 6)
    labels = np.zeros((6, 6), dtype=np.uint16)
    labels[1:3, 1:3] = 1
    labels[3:5, 3:5] = 2
    description = describe_image(image)
    table = region_feature_table(image, labels)
    assert description["shape"] == (6, 6)
    assert table["label"].tolist() == [1, 2]
    assert table["area"].tolist() == [4.0, 4.0]


def test_radial_profile_accounts_for_every_mask_pixel():
    image = np.ones((7, 7), dtype=np.float32)
    mask = np.zeros_like(image, dtype=bool)
    mask[1:6, 1:6] = True
    profile = normalized_radial_profile(image, mask, bins=4)
    assert profile["pixel_count"].sum() == int(mask.sum())
    assert profile.loc[profile.pixel_count > 0, "mean_intensity"].eq(1.0).all()
