from scdataloader import Preprocessor
from scdataloader.preprocess import additional_postprocess, LaminPreprocessor
import pandas as pd
import lamindb as ln
from upath import UPath
from anndata import AnnData, read_h5ad

col = ln.Collection.filter(key__contains="scbasecamp").first()

# for i, val in enumerate(col.artifacts.filter()):
#    if i > 0: # to continue from the previous stopped point
#        val.cache()
#        print(i)
#
mdf = pd.read_parquet("notebooks/data/tahoe_final_metadata.parquet")
mdf = mdf.set_index("srx_accession")


def preprocess(adata):
    elem = mdf.loc[adata.obs["SRX_accession"].iloc[0]]
    adata.obs["organism_ontology_term_id"] = elem.organism_ontology_term_id
    adata.obs["tissue_ontology_term_id"] = elem.tissue_ontology_term_id
    adata.obs["disease_ontology_term_id"] = elem.disease_ontology_term_id
    adata.obs["assay_ontology_term_id"] = elem.assay_ontology_term_id
    adata.obs["cell_type"] = "unknown"
    adata.obs["cell_type_ontology_term_id"] = "unknown"
    adata.obs["sex"] = "unknown"
    adata.obs["sex_ontology_term_id"] = "unknown"
    adata.obs["age"] = "unknown"
    adata.obs["age_group"] = "unknown"
    adata.obs["development_stage_ontology_term_id"] = "unknown"
    adata.obs["self_reported_ethnicity_ontology_term_id"] = "unknown"
    adata.obs["ethnicity"] = "unknown"
    adata.obs["is_primary_data"] = True
    adata.obs["cell_culture"] = "True"
    if "gene_ids" in adata.var.columns:
        adata.var.symbol = adata.var.index
        adata.var["ensembl_ontology_term_id"] = adata.var["gene_ids"]
        adata.var = adata.var.set_index("gene_ids")
    return adata


def cache_path(artifact):
    cloud_path = UPath(artifact.storage.root) / artifact.key
    cache_path = ln.setup.settings.paths.cloud_to_local_no_update(cloud_path)
    return cache_path


preprocessor = Preprocessor(
    is_symbol=False,
    keepdata=True,  # velocyto stuff to keep
    organisms=list(set(mdf.organism_ontology_term_id.unique())),
    additional_preprocess=preprocess,
    additional_postprocess=additional_postprocess,
)

f = [
    int(i.split(" p")[-1])
    for i in ln.Artifact.filter(description__contains="scbasecamp", version="2")
    .df()
    .description
]
print(len(f))
unprocessed = list(set(list(range(27253))) - set(f))
unprocessed.sort()
print(len(unprocessed))
elems = col.artifacts.filter()
for n, i in enumerate(unprocessed[1700:1800]):
    print(n, i)
    path = cache_path(elems[i])
    adata = read_h5ad(path)
    print(adata)
    try:
        adata = preprocessor(adata, dataset_id=elems[i].stem_uid)
        print("done preprocess")
        myfile = ln.Artifact.from_anndata(
            adata,
            description="scbasecamp preprocessed p" + str(i),
            version="2",
        )
        myfile.save()
        del myfile
        del adata
    except Exception as e:
        print(e)
