import pandas as pd
import numpy as np
import requests
import zipfile
import os

cp_input_dir = "cp_input"

###############################################
#  Select example wells/plates from metadata  #
###############################################

# fetch metadata
meta_url = "https://data.broadinstitute.org/bbbc/BBBC022/BBBC022_v1_image.csv"
meta = pd.read_csv(meta_url, on_bad_lines="skip", low_memory=False)

# get metadata of mock and cytoD-treated plates/wells
mock_meta = meta.query('Image_Metadata_ASSAY_WELL_ROLE == "mock"')
cyto_meta = meta.query('Image_Metadata_SOURCE_COMPOUND_NAME == "cytochalasin D"')

# use one cytoD well as an example
cyto_row = cyto_meta[["Image_Metadata_CPD_WELL_POSITION", "Image_Metadata_PlateID"]].drop_duplicates().head(1)
cyto_meta = pd.merge(cyto_row, cyto_meta)

# select mock wells in the same plate with cytoD
mock_meta = mock_meta[mock_meta.Image_Metadata_PlateID.isin(cyto_meta.Image_Metadata_PlateID)]

# use one mock well as an example
mock_row = mock_meta[["Image_Metadata_CPD_WELL_POSITION", "Image_Metadata_PlateID"]].drop_duplicates().head(1)
mock_meta = pd.merge(mock_row, mock_meta)

# combine metadata of mock and cytoD
mock_cyto_meta = pd.concat([mock_meta, cyto_meta])

# accomodate image path to zip file names
plate_ids = mock_cyto_meta.Image_Metadata_PlateID
mock_cyto_meta.Image_PathName_OrigER = plate_ids.apply(lambda plate_id: f"BBBC022_v1_images_{plate_id}w2/")
mock_cyto_meta.Image_PathName_OrigHoechst = plate_ids.apply(lambda plate_id: f"BBBC022_v1_images_{plate_id}w1/")
mock_cyto_meta.Image_PathName_OrigMito = plate_ids.apply(lambda plate_id: f"BBBC022_v1_images_{plate_id}w5/")
mock_cyto_meta.Image_PathName_OrigPh_golgi = plate_ids.apply(lambda plate_id: f"BBBC022_v1_images_{plate_id}w4/")
mock_cyto_meta.Image_PathName_OrigSyto = plate_ids.apply(lambda plate_id: f"BBBC022_v1_images_{plate_id}w3/")

# save metadata
os.makedirs(cp_input_dir, exist_ok=True)
meta_csv = os.path.join(cp_input_dir, "metadata.csv")
mock_cyto_meta.to_csv(meta_csv, index=False)


####################################
#       Download image files       #
####################################

# get URLs of image files
img_url = "https://data.broadinstitute.org/bbbc/BBBC022/BBBC022_v1_images_urls.txt"
img = pd.read_csv(img_url, header=None, names=['url'])

# filter image URLs to include only necessary ones
img['plate_id'] = img.url.str.extract("BBBC022_v1_images_(\d+)w\d+.zip").astype(int)
img = img[img.plate_id.isin(plate_ids)]

# download necessary images
for idx, row in img.iterrows():
    # download zip
    img_zip = os.path.join(cp_input_dir, os.path.basename(row.url))
    with requests.get(row.url, stream=True) as r:
        r.raise_for_status()
        with open(img_zip, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    # unzip downloaded
    with zipfile.ZipFile(img_zip, "r") as z:
        z.extractall(cp_input_dir)
