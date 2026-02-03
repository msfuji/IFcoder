# IFcoder

**IFcoder** is a Python package for embedding immunofluorescence cell images 
into the latent space using variational autoencoders (VAEs).

It is designed to operate seamlessly with [**CellProfiler**](https://cellprofiler.org/) 
and [**Scanpy**](https://scanpy.readthedocs.io/),
making it easy to move from image segmentation to downstream single-cell analysis.

---

## Key features

- Extracts image patches centered at cell objects based on CellProfiler segmentation.
- Learns embeddings of the image patches using a convolutional VAE.
- Stores the embeddings in a single [**AnnData**](https://anndata.readthedocs.io/) file
along with the image patches, CellProfiler measurements, and metadata.

---

## Installation
```
pip install ifcoder
```

## Input
- CellProfiler output CSV. Four columns `Location_Center_X`, `Location_Center_Y`, `ImageNumber`, `ObjectNumber` 
are mandatory. Optionally, user can append additional columns of metadata (e.g., batch, treatment, diagnosis).
- Input images of the CellProfiler pipeline. Image patches will be extracted from them.

## Example
please see [ifcoder_example.ipynb](xxx.html).

## Usage
1. **Extract image patches based on CellProfiler output**
```
ifcoder extract --cp-csv ${cellprofiler_csv} --out ${patches_h5ad}
```
This step:
- reads a CellProfiler output (e.g. Cells.csv),
- extracts image patches centered on cells,
- saves the image patches, metadata, and measurements in an AnnData file

2. **Train VAE and compute embeddings**
```
ifcoder train --data ${patches_h5ad} --out ${embeddings_h5ad}
```
This runs VAE and adds learned embeddings to the AnnData object
`adata.obsm["X_ifcoder"]`

3. **Downstream analysis**
```
import scanpy as sc

adata = sc.read_h5ad("embeddings.h5ad")
sc.pp.neighbors(adata, use_rep="X_ifcoder")
sc.tl.umap(adata)
sc.pl.umap(adata)
```
This example draws UMAP dimentionality reduction of cells.

## Output
IFcoder uses [**AnnData**](https://anndata.readthedocs.io/) as the central data structure:
- `adata.obsm["X_ifcoder"]`

    Learned VAE embeddings of cell objects.

- `adata.obsm["patches"]`

    Image patches of cell objects with dimensions (n_cells, n_channels, height, width).

- `adata.X`

    CellProfiler quantitative measurements of cell objects.
    The columns starting with `AreaShape_` and `Intensity_` in CellProfiler output are stored here.

- `adata.obs`

    Cell metadata (the columns without prefixes `AreaShape_` and `Intensity_`).