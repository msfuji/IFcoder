from __future__ import annotations

from pathlib import Path

import pandas as pd
import tifffile as tiff
import pytest


DATA_DIR = Path(__file__).resolve().parent / "data"
CSV_NAME = "Cells.csv"


def _prepare_cp_csv(tmp_path: Path, patch_size: int) -> Path:
    """Return a temporary CSV with local PathName_* values and in-bounds rows."""
    src_csv = DATA_DIR / CSV_NAME
    if not src_csv.exists():
        pytest.skip(f"missing test CSV at {src_csv}")

    df = pd.read_csv(src_csv)

    # Update PathName_* columns to local tests/data directory
    for col in df.columns:
        if col.startswith("PathName_"):
            df[col] = str(DATA_DIR) + "/"

    # Read one image to get dimensions for edge filtering
    fname_cols = [c for c in df.columns if c.startswith("FileName_")]
    if not fname_cols:
        pytest.skip("no FileName_* columns found in CSV")

    first_fname = df.loc[0, fname_cols[0]]
    img_path = DATA_DIR / first_fname
    if not img_path.exists():
        pytest.skip(f"missing image {img_path}")

    img = tiff.imread(img_path)
    if img.ndim != 2:
        pytest.skip("expected 2D grayscale TIFF for tests")

    h, w = img.shape
    r = patch_size // 2

    # Keep only rows with centers inside the image bounds
    mask = (
        (df["Location_Center_X"] >= r)
        & (df["Location_Center_X"] < w - r)
        & (df["Location_Center_Y"] >= r)
        & (df["Location_Center_Y"] < h - r)
    )
    df = df.loc[mask].reset_index(drop=True)

    if len(df) == 0:
        pytest.skip("no in-bounds rows for selected patch size")

    out_csv = tmp_path / "Cells_local.csv"
    df.to_csv(out_csv, index=False)
    return out_csv


@pytest.fixture()
def cp_csv(tmp_path: Path) -> Path:
    return _prepare_cp_csv(tmp_path, patch_size=32)
