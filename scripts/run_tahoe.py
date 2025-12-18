import pandas as pd
import lamindb as ln
from scdataloader.preprocess import LaminPreprocessor, additional_postprocess

df = pd.read_csv("data/tahoe_metadata.tsv", sep="\t")

col = ln.Collection.filter(name="tahoe").one()


def additional_preprocess(adata):
    # Map df columns to adata.obs using cell_name as the key
    for col in df.columns[1:]:
        adata.obs[col] = adata.obs["cell_name"].map(df.set_index("cell_name")[col])
    adata.obs["cell_culture"] = "True"
    adata.obs["organism_ontology_term_id"] = "NCBITaxon:9606"
    adata.obs["is_primary_data"] = True
    adata.obs["cell_type"] = "unknown"
    return adata


preprocess = LaminPreprocessor(
    organisms=["NCBITaxon:9606"],
    keepdata=False,
    use_raw=True,
    additional_preprocess=additional_preprocess,
    is_symbol=True,
    additional_postprocess=additional_postprocess,
    keep_files=False,
)

col = preprocess(col, name="tahoe_preprocessed", start_at=10)
