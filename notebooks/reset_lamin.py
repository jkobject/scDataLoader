#only used by me to reset the lamin storage info when I am copying my data from one
# location to another

import anndata as ad
import lamindb as ln
from scdataloader import DataModule
from lamindb import settings

description="preprocessed for scprint"

a = !ls zhang2024_adata_*.h5ad
for f in a:
    adata = f.load(stream=True) #ad.read_h5ad(f)
    adata.obs['cell_type_ontology_term_id'] = adata.obs['cell_type_ontology_term_id'].replace(rn)
    adata.obs['cell_culture'] = False
    ! rm $f.path
    adata.write(f.path)
    ln.Artifact(adata, description=description).save()
    print(f)
# this dataset alone is 13M cells
ln.Storage.filter(uid="GZgLW1TI").update(root=settings.storage)
files = ln.Artifact.filter(description=description)
len(files)
dataset = ln.Collection(files, name="temp2", description="temp for the scprint preprocessing")
dataset.save()
datamodule = DataModule(collection_name="temp2")

