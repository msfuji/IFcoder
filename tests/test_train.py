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


def test_anndata_patches_invalid_normalize(cp_csv: Path, tmp_path: Path) -> None:
    h5ad_path = _make_h5ad(tmp_path, cp_csv)

    with pytest.raises(ValueError, match="normalize must be one of"):
        AnnDataPatches(str(h5ad_path), normalize="bad_value")
