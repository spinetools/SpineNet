"""Tests for extracting IVD volumes at a configurable shape and saving them
(no weights needed)."""

import os

import nibabel as nib
import numpy as np
import pytest

from spinenet import SpineNet
from spinenet.io import save_ivd_volumes


@pytest.fixture
def ivd_dicts():
    """Synthetic IVDs shaped like SpineNet.get_ivds_from_vert_dicts output."""
    rng = np.random.default_rng(0)
    return [
        {"volume": rng.random((12, 128, 256)), "level_name": name}
        for name in ["L5-S1", "L4-L5", "L3-L4"]
    ]


def test_save_keeps_volume(ivd_dicts, tmp_path):
    paths = save_ivd_volumes(ivd_dicts, str(tmp_path / "ivds"))
    assert [os.path.basename(p) for p in paths] == [
        "L5-S1.npy",
        "L4-L5.npy",
        "L3-L4.npy",
    ]
    for path, ivd_dict in zip(paths, ivd_dicts):
        vol = np.load(path)
        assert vol.dtype == np.float32
        np.testing.assert_array_equal(
            vol, ivd_dict["volume"].astype(np.float32)
        )


def test_nifti_and_prefix(ivd_dicts, tmp_path):
    paths = save_ivd_volumes(
        ivd_dicts, str(tmp_path), file_format="nii.gz", prefix="scan1_"
    )
    assert paths[0].endswith("scan1_L5-S1.nii.gz")
    assert nib.load(paths[0]).get_fdata().shape == (12, 128, 256)


def test_invalid_inputs(ivd_dicts, tmp_path):
    with pytest.raises(KeyError):
        save_ivd_volumes([{"volume": np.zeros((9, 112, 224))}], str(tmp_path))
    with pytest.raises(ValueError):
        save_ivd_volumes(ivd_dicts, str(tmp_path), file_format="png")
    with pytest.raises(ValueError):
        save_ivd_volumes(ivd_dicts * 2, str(tmp_path))


def synthetic_vert_dicts(cx, bottom_cy, n_slices=12):
    """Four axis-aligned 80x60 px vertebrae stacked up from bottom_cy."""
    vert_dicts = []
    for i, label in enumerate(["S1", "L5", "L4", "L3"]):
        cy = bottom_cy - i * 76
        poly = [
            [cx + 40, cy - 30],
            [cx + 40, cy + 30],
            [cx - 40, cy + 30],
            [cx - 40, cy - 30],
        ]
        vert_dicts.append(
            {
                "predicted_label": label,
                "average_polygon": poly,
                "polys": [poly] * n_slices,
                "slice_nos": list(range(n_slices)),
            }
        )
    return vert_dicts


def extract(cx=256, bottom_cy=300, n_slices=12, **kwargs):
    scan = np.random.default_rng(0).uniform(100, 200, (512, 512, n_slices))
    spnt = SpineNet.__new__(SpineNet)  # IVD extraction needs no weights
    return spnt.get_ivds_from_vert_dicts(
        synthetic_vert_dicts(cx, bottom_cy, n_slices), scan, **kwargs
    )


# centre of scan, lowest disc near the bottom edge, discs near the right edge
@pytest.mark.parametrize("cx, bottom_cy", [(256, 300), (256, 490), (440, 300)])
def test_bigger_crop_contains_default(tmp_path, cx, bottom_cy):
    default = extract(cx, bottom_cy)
    big = extract(cx, bottom_cy, output_shape=(12, 128, 256))
    assert [d["level_name"] for d in big] == ["L5-S1", "L4-L5", "L3-L4"]
    for small_dict, big_dict in zip(default, big):
        assert small_dict["volume"].shape == (9, 112, 224)
        assert big_dict["volume"].shape == (12, 128, 256)
        # same scale and slices, just more context around the default crop
        np.testing.assert_array_equal(
            big_dict["volume"][2:11, 8:120, 16:240], small_dict["volume"]
        )
    paths = save_ivd_volumes(big, str(tmp_path))
    assert all(np.load(p).shape == (12, 128, 256) for p in paths)


def test_only_real_slices():
    # 6 scan slices into 12 output slices: 6 real ones, the rest zero padded
    for ivd_dict in extract(n_slices=6, output_shape=(12, 128, 256)):
        filled = ivd_dict["volume"].reshape(12, -1).any(axis=1)
        assert filled.sum() == 6


def test_crop_bigger_than_patch():
    for ivd_dict in extract(output_shape=(15, 256, 400)):
        assert ivd_dict["volume"].shape == (15, 256, 400)
        assert np.isfinite(ivd_dict["volume"]).all()


def test_invalid_output_shape_and_grading_guard():
    with pytest.raises(ValueError):
        extract(output_shape=(128, 256))
    with pytest.raises(ValueError):
        extract(output_shape=(0, 128, 256))
    big = extract(output_shape=(12, 128, 256))
    with pytest.raises(ValueError, match="9, 112, 224"):
        SpineNet.__new__(SpineNet).grade_ivds(big)
