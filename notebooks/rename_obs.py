import lamindb as ln
import os

pos = 10819  # 824
for p, i in enumerate(
    ln.Collection.filter(key="scPRINT-V2 (all+tahoe+scbase)")
    .one()
    .artifacts.filter()[pos:]
):
    print(p + pos)
    try:
        adata = i.open()
    except Exception as e:
        print(i.path)
        raise e
    if "cell_type_ontology_term_id" not in adata.obs.columns:
        print(i.path)
        print("cell type column not found")
        break
    adata = i.load()
    if "age_group" not in adata.obs.columns:
        print(i.path)
        adata.obs["age_group"] = "unknown"
    if "simplified_dev_stage" in adata.obs.columns:
        if (adata.obs["age_group"] != adata.obs["simplified_dev_stage"]).any():
            print(i.path)
            raise ValueError("age group mismatch")
    adata.obs["cell_culture"] = adata.obs["cell_culture"].astype(str)
    adata.write_h5ad(str(i.path) + ".tmp")
    os.rename(str(i.path) + ".tmp", str(i.path))

    # if (adata.obs["cell_culture"] != "unknown").any():
    #    print(i.stem_uid)
    #    adata = i.load()
    #    adata.obs["cell_culture"] = "unknown"
    #    adata.write_h5ad(i.path)
