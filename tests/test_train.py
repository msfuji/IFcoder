from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ifcoder.train import AnnDataPatches
from ifcoder.extract import extract


def _make_h5ad(tmp_path: Path, cp_csv: Path) -> Path:
    out = tmp_path / "patches.h5ad"
    extract(
        cp_csv=str(cp_csv),
        out_h5ad=str(out),
        patch_size=32,
    )
    return out


def test_anndata_patches_normalize(cp_csv: Path, tmp_path: Path) -> None:
    h5ad_path = _make_h5ad(tmp_path, cp_csv)

    ds = AnnDataPatches(str(h5ad_path), normalize="per_channel")
    assert len(ds) > 0

    x = ds[0]
    assert x.ndim == 3
    assert np.isfinite(x.numpy()).all()

    # Per-channel normalization should keep max <= 1
    x_all = ds.x.numpy()
    ch_max = x_all.reshape(x_all.shape[0], x_all.shape[1], -1).max(axis=(0, 2))
    assert (ch_max <= 1.0 + 1e-6).all()


def test_anndata_patches_augment_default_on(cp_csv: Path, tmp_path: Path) -> None:
    """Augmentation is on by default and changes output across calls."""
    h5ad_path = _make_h5ad(tmp_path, cp_csv)

    ds = AnnDataPatches(str(h5ad_path))
    assert ds.augment is True

    # Two calls should almost certainly differ (flips, noise, scaling)
    samples = [ds[0] for _ in range(10)]
    any_diff = any(not np.array_equal(samples[0].numpy(), s.numpy()) for s in samples[1:])
    assert any_diff, "augmented samples should vary across calls"


def test_anndata_patches_augment_off(cp_csv: Path, tmp_path: Path) -> None:
    """With augment=False, __getitem__ returns identical data each call."""
    h5ad_path = _make_h5ad(tmp_path, cp_csv)

    ds = AnnDataPatches(str(h5ad_path), augment=False)
    a = ds[0]
    b = ds[0]
    assert np.array_equal(a.numpy(), b.numpy())


def test_anndata_patches_augment_preserves_shape(cp_csv: Path, tmp_path: Path) -> None:
    h5ad_path = _make_h5ad(tmp_path, cp_csv)

    ds = AnnDataPatches(str(h5ad_path))
    raw_shape = ds.x[0].shape
    aug_shape = ds[0].shape
    assert raw_shape == aug_shape


def test_anndata_patches_invalid_normalize(cp_csv: Path, tmp_path: Path) -> None:
    h5ad_path = _make_h5ad(tmp_path, cp_csv)

    with pytest.raises(ValueError, match="normalize must be one of"):
        AnnDataPatches(str(h5ad_path), normalize="bad_value")
