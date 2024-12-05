# import lamindb as ln
# import numpy as np
import os
import time

import lamindb as ln
import pytest
import scanpy as sc
from torch import max as tmax
from torch.utils.data import DataLoader

from scdataloader import Collator, DataModule, Preprocessor, SimpleAnnDataset, utils
from scdataloader.base import NAME
from scdataloader.preprocess import additional_postprocess, additional_preprocess


def test_base():
    assert NAME == "scdataloader"
    adata = sc.read_h5ad(os.path.join(os.path.dirname(__file__), "test.h5ad"))
    print("populating ontology...")
    start_time = time.time()
    utils.populate_my_ontology(
        organisms=["NCBITaxon:10090", "NCBITaxon:9606"],
        sex=["PATO:0000384", "PATO:0000383"],
        # celltypes=None,
        # ethnicities=None,
        # assays=None,
        # tissues=None,
        # diseases=None,
        # dev_stages=None,
    )
    end_time = time.time()
    print(f"ontology populated in {end_time - start_time:.2f} seconds")
    preprocessor = Preprocessor(
        do_postp=True,
        additional_postprocess=additional_postprocess,
        additional_preprocess=additional_preprocess,
        force_preprocess=True,
    )
    adata = preprocessor(adata)
    art = ln.Artifact(adata, description="test")
    art.save()
    ln.Collection(art, name="test", description="test").save()
    datamodule = DataModule(
        collection_name="test",
        organisms=["NCBITaxon:9606"],  # organism that we will work on
        how="most expr",  # for the collator (most expr genes only will be selected)
        max_len=1000,  # only the 1000 most expressed
        batch_size=64,
        do_gene_pos=False,
        num_workers=1,
        use_default_col=True,
        clss_to_weight=["organism_ontology_term_id", "cell_type_ontology_term_id"],
        clss_to_predict=["organism_ontology_term_id", "cell_type_ontology_term_id"],
        all_clss=["organism_ontology_term_id", "cell_type_ontology_term_id"],
        hierarchical_clss=["cell_type_ontology_term_id"],
        validation_split=0.1,
        test_split=0,
    )
    datamodule.setup()
    for i in datamodule.train_dataloader():
        # pass #or do pass
        print(i)
        break
    assert i["x"][0, 0] >= i["x"][0, 1] >= i["x"][0, -1]
    assert tmax(i["class"][:, 1]) <= 15
