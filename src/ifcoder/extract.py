import os
import numpy as np
import pandas as pd
import tifffile as tiff
import anndata as ad


def _detect_channels(df: pd.DataFrame) -> list[str]:
    """
    Detect channel suffixes from CellProfiler columns:
      FileName_<CH> and PathName_<CH>

    Returns channels in the same order they appear in the CSV header.
    """
    file_cols = [c for c in df.columns if c.startswith("FileName_")]
    channels = [c[len("FileName_") :] for c in file_cols]

    # Keep only channels that also have PathName_<CH>
    channels = [ch for ch in channels if f"PathName_{ch}" in df.columns]

    if not channels:
        raise ValueError(
            "No channels detected. Expected FileName_<CH> and PathName_<CH> columns."
        )
    return channels


def _img_paths(row: pd.Series, channels: list[str]) -> list[str]:
    return [
        os.path.join(str(row[f"PathName_{ch}"]), str(row[f"FileName_{ch}"]))
        for ch in channels
    ]


def _load_stack(paths: list[str]) -> np.ndarray:
    imgs = []
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Image not found: {p}")
        img = tiff.imread(p)
        if img.ndim != 2:
            raise ValueError(
                f"Expected 2D grayscale TIFF, got shape {img.shape} at {p}"
            )
        imgs.append(img)
    return np.stack(imgs, axis=0)  # (C, H, W)


def _patch(stack: np.ndarray, x: float, y: float, ps: int) -> np.ndarray | None:
    r = ps // 2
    x = int(round(float(x)))
    y = int(round(float(y)))
    y0, y1 = y - r, y - r + ps
    x0, x1 = x - r, x - r + ps

    C, H, W = stack.shape
    if y0 < 0 or x0 < 0 or y1 > H or x1 > W:
        return None
    return stack[:, y0:y1, x0:x1]


def extract(
    cp_csv: str,
    out_h5ad: str,
    patch_size: int = 64,
    channels: list[str] | None = None,
) -> None:
    """
    Extract multi-channel image patches centered on nuclei based on a CellProfiler CSV
    and save as AnnData (.h5ad).

    AnnData layout:
      - X        : CellProfiler measurements (AreaShape_*, Intensity_*)
      - obs      : All remaining CP columns (metadata)
      - obsm     : patches (N, C, P, P)
      - obs_names: barcodes = ImageNumber_ObjectNumber
    """
    df = pd.read_csv(cp_csv)
    
    n_total = len(df)

    required = {
        "ImageNumber",
        "ObjectNumber",
        "Location_Center_X",
        "Location_Center_Y",
    }
    if not required <= set(df.columns):
        raise ValueError(f"CSV must contain columns: {required}")

    if channels is None:
        channels = _detect_channels(df)

    # ---------------------------------------
    # define measurement vs metadata columns
    # ---------------------------------------
    meas_cols = [
        c
        for c in df.columns
        if c.startswith("AreaShape_") or c.startswith("Intensity_")
    ]
    obs_cols = [c for c in df.columns if c not in meas_cols]

    # ---------------------------------------
    # grouping: load each image stack once
    # ---------------------------------------
    group_cols = []
    for ch in channels:
        group_cols += [f"PathName_{ch}", f"FileName_{ch}"]

    patches = []
    centers = []
    keep_rows = []

    for _, g in df.groupby(group_cols):
        paths = _img_paths(g.iloc[0], channels)
        stack = _load_stack(paths)

        for idx, row in g.iterrows():
            p = _patch(
                stack,
                row["Location_Center_X"],
                row["Location_Center_Y"],
                patch_size,
            )
            if p is None:
                continue

            patches.append(p)
            centers.append(
                (row["Location_Center_X"], row["Location_Center_Y"])
            )
            keep_rows.append(idx)

    if not patches:
        raise RuntimeError(
            "No patches extracted. Check patch_size and coordinate ranges."
        )

    n_kept = len(patches)
    n_dropped = n_total - n_kept

    # ---------------------------------------
    # subset dataframe to kept cells
    # ---------------------------------------
    df = df.loc[keep_rows].reset_index(drop=True)

    patches = np.stack(patches, axis=0).astype(np.float32)  # (N, C, P, P)
    centers = np.asarray(centers, dtype=np.float32)

    # ---------------------------------------
    # build AnnData
    # ---------------------------------------
    X = df[meas_cols].to_numpy(dtype=np.float32)
    obs = df[obs_cols].copy()

    barcodes = (
        "img"
        + df["ImageNumber"].astype(str)
        + "_obj"
        + df["ObjectNumber"].astype(str)
    )
    obs.index = barcodes

    adata = ad.AnnData(X=X, obs=obs)
    adata.obs_names = barcodes
    adata.var_names = meas_cols

    adata.obsm["patches"] = patches
    adata.obsm["centers"] = centers

    adata.uns["ifcoder"] = {
        "patch_size": patch_size,
        "channels": channels,
        "source": "CellProfiler",
        "edge_filtered": True,
    }

    adata.write_h5ad(out_h5ad)

    print(f"Detected channels: {channels}")
    print(f"Saved AnnData -> {out_h5ad}")
    print(f"Cells in CSV        : {n_total}")
    print(f"Cells kept (patches): {n_kept}")
    print(f"Cells dropped (edge): {n_dropped} "
          f"({n_dropped / n_total:.1%})")
    print(f"Measurements: {adata.n_vars}")
    print(f"Dimensions of patches: {patches.shape}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Extract multi-channel nucleus-centered patches from a "
            "CellProfiler Nuclei.csv and save as AnnData (.h5ad)"
        )
    )
    parser.add_argument("--cp-csv", required=True, help="Path to CellProfiler Nuclei.csv")
    parser.add_argument("--out", required=True, help="Output .h5ad path")
    parser.add_argument(
        "--patch-size", type=int, default=64, help="Patch size (pixels), default=64"
    )
    parser.add_argument(
        "--channels",
        nargs="*",
        default=None,
        help=(
            "Optional list of channel suffixes (after FileName_/PathName_). "
            "If omitted, auto-detect from CSV."
        ),
    )

    args = parser.parse_args()

    extract(
        cp_csv=args.cp_csv,
        out_h5ad=args.out,
        patch_size=args.patch_size,
        channels=args.channels,
    )
