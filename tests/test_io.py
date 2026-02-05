from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from ifcoder.io import export_anndata


def test_export_anndata(tmp_path: Path) -> None:
    emb = np.random.RandomState(0).randn(5, 3).astype(np.float32)
    obs = pd.DataFrame({"batch": ["a", "a", "b", "b", "c"]})

    out = tmp_path / "embeddings.h5ad"
    export_anndata(embedding=emb, out=str(out), obs=obs, obsm_key="X_test", uns={"k": 1})

    adata = ad.read_h5ad(out)
    assert adata.obsm["X_test"].shape == (5, 3)
    assert "ifcoder" in adata.uns
    assert adata.uns["ifcoder"]["k"] == 1
    assert list(adata.obs["batch"]) == ["a", "a", "b", "b", "c"]
