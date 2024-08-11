import lamindb as ln
import scanpy as sc
import numpy as np
import os

from scdataloader.base import NAME
from scdataloader import utils

from scdataloader import DataModule
from scdataloader import SimpleAnnDataset
from scdataloader import Collator
from scdataloader import Preprocessor

from torch.utils.data import DataLoader


def test_base():
    assert NAME == "scdataloader"
    utils.populate_my_ontology(
        organisms=["NCBITaxon:10090", "NCBITaxon:9606"],
        sex=["PATO:0000384", "PATO:0000383"],
        celltypes=None,
        ethnicities=None,
        assays=None,
        tissues=None,
        diseases=None,
        dev_stages=None,
    )
    # cx_dataset = (
    #     ln.Collection.using(instance="laminlabs/cellxgene")
    #     .filter(name="cellxgene-census", version="2023-12-15")
    #     .one()
    # )
    # datamodule = DataModule(
    #     collection_name="preprocessed dataset",
    #     organisms=["NCBITaxon:9606"],  # organism that we will work on
    #     how="most expr",  # for the collator (most expr genes only will be selected)
    #     max_len=1000,  # only the 1000 most expressed
    #     batch_size=64,
    #     num_workers=1,
    #     validation_split=0.1,
    #     test_split=0,
    # )
    # for i in datamodule.train_dataloader():
    #     # pass #or do pass
    #     print(i)
    #     break
    # assert True, "Datamodule test passed"

    adata = sc.read_h5ad(os.path.join(os.path.dirname(__file__), "test.h5ad"))
    preprocessor = Preprocessor(do_postp=False)
    adata = preprocessor(adata)

    adataset = SimpleAnnDataset(adata, obs_to_output=["organism_ontology_term_id"])
    col = Collator(
        organisms=["NCBITaxon:9606"],
        max_len=1000,
        how="random expr",
    )
    dataloader = DataLoader(
        adataset,
        collate_fn=col,
        batch_size=4,
        num_workers=1,
        shuffle=False,
    )
    for batch in dataloader:
        print(batch)
        break
    assert True, "SimpleAnnDataset test passed"
