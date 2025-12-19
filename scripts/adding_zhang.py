# only used by me to reset the lamin storage info when I am copying my data from one
# location to another

import anndata as ad
import lamindb as ln
from lamindb import settings

from scdataloader import DataModule

description = "ready for preprocessing"
# Use glob to find files matching the pattern
import glob
import os

files = glob.glob("zhang2024_adata_*.h5ad")
for f in files:
    adata = ad.read_h5ad(f)
    # Replace 'rn' with the actual mapping or value you want to use
    adata.obs["cell_type_ontology_term_id"] = adata.obs[
        "cell_type_ontology_term_id"
    ].replace(rn)
    adata.obs["cell_culture"] = False
    os.remove(f)
    adata.write(f)
    ln.Artifact(adata, description=description).save()
    print(f)
# this dataset alone is 13M cells
files = ln.Artifact.filter(description=description)
dataset = ln.Collection(
    files, key="temp2", description="temp for the scprint preprocessing"
)
dataset.save()
