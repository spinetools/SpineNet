"""Tests for extracting IVD volumes at a configurable shape and saving them
(no weights needed)."""

import os

import nibabel as nib
import numpy as np
import pytest

from spinenet import SpineNet
from spinenet.io import save_ivd_volumes
from spinenet.utils import gen_utils


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
    # a scan affine needs the 'voxel_to_scan' of each IVD
    with pytest.raises(KeyError):
        save_ivd_volumes(
            ivd_dicts, str(tmp_path), "nii.gz", scan_affine=np.eye(4)
        )


def synthetic_vert_dicts(cx, bottom_cy, n_slices=12, tilt=0):
    """Four 80x60 px vertebrae stacked up from bottom_cy, axis-aligned or
    rotated by tilt degrees about the lowest one."""
    cos, sin = np.cos(np.deg2rad(tilt)), np.sin(np.deg2rad(tilt))
    vert_dicts = []
    for i, label in enumerate(["S1", "L5", "L4", "L3"]):
        cy = bottom_cy - i * 76
        poly = [
            [cx + 40, cy - 30],
            [cx + 40, cy + 30],
            [cx - 40, cy + 30],
            [cx - 40, cy - 30],
        ]
        if tilt:
            poly = [
                [
                    cx + (x - cx) * cos - (y - bottom_cy) * sin,
                    bottom_cy + (x - cx) * sin + (y - bottom_cy) * cos,
                ]
                for x, y in poly
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


@pytest.mark.parametrize("tilt", [0, 15, -25])
@pytest.mark.parametrize(
    "output_shape", [(9, 112, 224), (12, 128, 256), (9, 200, 330)]
)
def test_voxel_to_scan_follows_extraction(monkeypatch, tilt, output_shape):
    # fixed vertebra median, so that volume == scan / 1000 without clipping
    monkeypatch.setattr(
        gen_utils,
        "get_vbs_intensity",
        lambda volume, all_vb_x, *args: [np.full(1, 250.0)] * 4,
    )
    spnt = SpineNet.__new__(SpineNet)
    vert_dicts = synthetic_vert_dicts(256, 330, n_slices=14, tilt=tilt)
    # (slice, row, col, 1) of every voxel of an IVD volume
    index = np.stack([*np.indices(output_shape), np.ones(output_shape)], -1)
    # scans that hold their own row, column and slice index
    for axis, coordinate in enumerate(np.indices((448, 512, 14))):
        ivd_dicts = spnt.get_ivds_from_vert_dicts(
            vert_dicts, 200.0 + coordinate, output_shape=output_shape
        )
        assert len(ivd_dicts) == 3
        for ivd_dict in ivd_dicts:
            in_scan = (index @ ivd_dict["voxel_to_scan"].T)[..., axis]
            error = ivd_dict["volume"] * 1000 - 200 - in_scan
            # cv2 cubic interpolation is only exact to ~0.1 px on a ramp
            assert np.abs(error).max() < 0.2
            assert abs(error.mean()) < 0.01


def test_nifti_affine(tmp_path):
    ivd_dicts = extract(output_shape=(12, 128, 256))
    # rows -> inferior, columns -> posterior, slices -> right, in mm
    scan_affine = np.array(
        [[0, 0, 4, -20], [0, -0.5, 0, 90], [-0.5, 0, 0, 120], [0, 0, 0, 1.0]]
    )
    paths = save_ivd_volumes(
        ivd_dicts, str(tmp_path), "nii.gz", scan_affine=scan_affine
    )
    for path, ivd_dict in zip(paths, ivd_dicts):
        img = nib.load(path)
        affine = scan_affine @ ivd_dict["voxel_to_scan"]
        np.testing.assert_allclose(img.affine, affine, atol=1e-4)
        assert nib.aff2axcodes(img.affine) == ("R", "I", "P")
        # 4 mm slices; the 80 px wide discs are scaled to 113.5 px, at 0.5 mm
        np.testing.assert_allclose(
            img.header.get_zooms(),
            (4, 0.5 * 80 / 113.5, 0.5 * 80 / 113.5),
            rtol=0.01,
        )
    # without a scan affine, the header keeps the identity
    path = save_ivd_volumes(ivd_dicts[:1], str(tmp_path / "vox"), "nii.gz")[0]
    np.testing.assert_array_equal(nib.load(path).affine, np.eye(4))

