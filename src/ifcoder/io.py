# src/ifcoder/io.py

import anndata as ad
import numpy as np
import pandas as pd
from typing import Mapping, Any, Optional


def export_anndata(
    embedding: np.ndarray,
    out: str,
    obs: Optional[pd.DataFrame] = None,
    obsm_key: str = "X_ifcoder",
    uns: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Export IFcoder embeddings to AnnData (.h5ad).

    Parameters
    ----------
    embedding
        Array of shape (n_obs, n_latent).
    out
        Output filename (.h5ad).
    obs
        Optional per-cell metadata (n_obs rows).
    obsm_key
        Key for embedding in adata.obsm (Scanpy convention: "X_*").
    uns
        Optional metadata stored under adata.uns["ifcoder"].
    """
    emb = np.asarray(embedding)
    if emb.ndim != 2:
        raise ValueError(f"embedding must be 2D (n_obs, n_latent), got {emb.shape}")

    n = emb.shape[0]

    if obs is None:
        obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n)])
    else:
        if len(obs) != n:
            raise ValueError(f"obs must have {n} rows to match embedding")
        obs = obs.copy()

    # X intentionally empty; embeddings live in obsm
    adata = ad.AnnData(
        X=np.zeros((n, 0), dtype=np.float32),
        obs=obs,
    )
    adata.obsm[obsm_key] = emb.astype(np.float32, copy=False)

    if uns is not None:
        adata.uns["ifcoder"] = dict(uns)

    adata.write_h5ad(out)
