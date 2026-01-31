import pandas as pd

# cellprofiler output for nuclei
nuc_file = "examples/cp_output/Nuclei.csv"
nuc = pd.read_csv(nuc_file)

# metadata of plate/well
meta_file = "examples/cp_input/metadata.csv"
meta = pd.read_csv(meta_file)

# merge based on Hoechst file name
df = pd.merge(nuc, meta, how = 'inner', 
              left_on = 'FileName_OrigHoechst', 
              right_on = 'Image_FileName_OrigHoechst',
              suffixes=('', '_original'))

# save CSV
outfile = "examples/cp_output/Nuclei_with_metadata.csv"
df.to_csv(outfile, index=False)
