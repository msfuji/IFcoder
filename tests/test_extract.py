from __future__ import annotations

from pathlib import Path

import anndata as ad
import pandas as pd
import pytest

from ifcoder.extract import extract


DATA_DIR = Path(__file__).resolve().parent / "data"


def test_extract_creates_h5ad(cp_csv: Path, tmp_path: Path) -> None:
    out = tmp_path / "patches.h5ad"

    extract(
        cp_csv=str(cp_csv),
        out_h5ad=str(out),
        patch_size=32,
    )

    assert out.exists()

    adata = ad.read_h5ad(out)
    assert "patches" in adata.obsm
    assert "centers" in adata.obsm
    assert adata.obsm["patches"].ndim == 4
    assert adata.obsm["centers"].shape[1] == 2
    assert "ifcoder" in adata.uns
    assert adata.uns["ifcoder"]["patch_size"] == 32
    assert adata.obsm["patches"].shape[0] == adata.n_obs


def test_extract_missing_required_columns(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "ImageNumber": [1],
            "ObjectNumber": [1],
            # Missing Location_Center_X / Location_Center_Y
        }
    )
    csv_path = tmp_path / "bad.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="CSV must contain columns"):
        extract(cp_csv=str(csv_path), out_h5ad=str(tmp_path / "out.h5ad"))


def test_extract_missing_image_raises(tmp_path: Path) -> None:
    src = DATA_DIR / "Cells.csv"
    if not src.exists():
        pytest.skip("missing test CSV")

    df = pd.read_csv(src).head(1)
    for col in df.columns:
        if col.startswith("PathName_"):
            df[col] = str(DATA_DIR) + "/"

    # Force a bad image filename
    for col in df.columns:
        if col.startswith("FileName_"):
            df[col] = "does_not_exist.tif"

    out_csv = tmp_path / "bad_image.csv"
    df.to_csv(out_csv, index=False)

    with pytest.raises(FileNotFoundError, match="Image not found"):
        extract(cp_csv=str(out_csv), out_h5ad=str(tmp_path / "out.h5ad"))
