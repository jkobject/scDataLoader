import lamindb as ln

for i in (
    ln.Collection.filter(key="scPRINT-V2 (all+tahoe+scbase)")
    .one()
    .artifacts.filter(description__contains="preprocessed dataset using")[:]
):
    try:
        adata = i.open()
    except Exception as e:
        print(i.path)
        raise e
    if "cell_type_ontology_term_id" not in adata.obs.columns:
        print(i.path)
        print("cell type column not found")
        break
    # if (adata.obs["cell_culture"] != "unknown").any():
    #    print(i.stem_uid)
    #    adata = i.load()
    #    adata.obs["cell_culture"] = "unknown"
    #    adata.write_h5ad(i.path)
