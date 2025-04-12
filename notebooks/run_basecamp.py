from scdataloader import Preprocessor
from scdataloader.preprocess import additional_postprocess, LaminPreprocessor
import pandas as pd
import lamindb as ln


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
    adata.obs["development_stage_ontology_term_id"] = "unknown"
    adata.obs["self_reported_ethnicity_ontology_term_id"] = "unknown"
    adata.obs["ethnicity"] = "unknown"
    adata.obs["is_primary_data"] = True
    adata.obs["cell_culture"] = True
    if "gene_ids" in adata.var.columns:
        adata.var.symbol = adata.var.index
        adata.var["ensembl_ontology_term_id"] = adata.var["gene_ids"]
        adata.var = adata.var.set_index("gene_ids")
    return adata


preprocessor = LaminPreprocessor(
    is_symbol=False,
    keepdata=True,  # velocyto stuff to keep
    organisms=list(set(mdf.organism_ontology_term_id.unique())),
    additional_preprocess=preprocess,
    additional_postprocess=additional_postprocess,
    force_lamin_cache=True,  # we have loaded them already (see comments above)
    keep_files=False,  # we will reach a memory issue otherwise (only works for small sets of data)
)

preprocessor(
    col,
    name="preprocessed dataset",
    description="scbasecamp preprocessed dataset",
    start_at=10976,  # to continue from the previous stopped point
    version="2",
)
