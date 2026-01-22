import os
import numpy as np
import pandas as pd
import tifffile as tiff


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
        raise ValueError("No channels detected. Expected FileName_<CH> and PathName_<CH> columns.")
    return channels


def _img_paths(row: pd.Series, channels: list[str]) -> list[str]:
    return [os.path.join(str(row[f"PathName_{ch}"]), str(row[f"FileName_{ch}"])) for ch in channels]


def _load_stack(paths: list[str]) -> np.ndarray:
    imgs = []
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Image not found: {p}")
        img = tiff.imread(p)
        if img.ndim != 2:
            raise ValueError(f"Expected 2D grayscale TIFF, got shape {img.shape} at {p}")
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
    out_npz: str,
    patch_size: int = 64,
    channels: list[str] | None = None,
) -> None:
    """
    Extract multi-channel image patches centered on nuclei based on a CellProfiler CSV

    Required columns in the CP CSV:
      - Location_Center_X
      - Location_Center_Y
      - FileName_<CH> and PathName_<CH> for each channel (channel names can vary)

    Saves NPZ keys:
      - patches: (N, C, P, P)
      - channels: (C,) channel suffixes (strings)
      - centers: (N, 2) float32 [x, y]
      - image_number: (N,) a lightweight identifier for the source image group
    """
    df = pd.read_csv(cp_csv)

    if "Location_Center_X" not in df.columns or "Location_Center_Y" not in df.columns:
        raise ValueError("CSV must contain Location_Center_X and Location_Center_Y columns.")

    if channels is None:
        channels = _detect_channels(df)

    # Group by image identity so we load each multi-channel image once
    group_cols = []
    for ch in channels:
        group_cols += [f"PathName_{ch}", f"FileName_{ch}"]

    patches = []
    centers = []
    image_numbers = []

    for _, g in df.groupby(group_cols):
        # Load image stack once for this group
        paths = _img_paths(g.iloc[0], channels)
        stack = _load_stack(paths)

        # Use a stable "image key" for metadata (first channel filename is enough)
        key = os.path.basename(paths[0])

        for x, y in zip(g["Location_Center_X"].values, g["Location_Center_Y"].values):
            p = _patch(stack, x, y, patch_size)
            if p is None:
                continue
            patches.append(p)
            centers.append((float(x), float(y)))
            image_numbers.append(key)

    if not patches:
        raise RuntimeError("No patches extracted. Check patch_size and coordinate ranges.")

    patches = np.stack(patches, axis=0)  # (N, C, P, P)

    np.savez_compressed(
        out_npz,
        patches=patches,
        channels=np.array(channels, dtype=object),
        centers=np.array(centers, dtype=np.float32),
        image_number=np.array(image_numbers, dtype=object),
    )

    print(f"Detected channels: {channels}")
    print(f"Saved {patches.shape[0]} patches (C={patches.shape[1]}, P={patches.shape[2]}) -> {out_npz}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract multi-channel nucleus-centered patches from a CellProfiler Nuclei.csv"
    )
    parser.add_argument("--cp-csv", required=True, help="Path to CellProfiler Nuclei.csv")
    parser.add_argument("--out", required=True, help="Output .npz path")
    parser.add_argument("--patch-size", type=int, default=64, help="Patch size (pixels), default=64")
    parser.add_argument(
        "--channels",
        nargs="*",
        default=None,
        help="Optional list of channel suffixes (after FileName_/PathName_). "
             "If omitted, auto-detect from CSV.",
    )

    args = parser.parse_args()

    extract(
        cp_csv=args.cp_csv,
        out_npz=args.out,
        patch_size=args.patch_size,
        channels=args.channels,
    )
