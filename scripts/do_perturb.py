import gc
import time

import lamindb as ln
import numpy as np
import pandas as pd
from anndata import AnnData, read_h5ad
from django.db.utils import OperationalError
from upath import UPath

from scdataloader import Preprocessor
from scdataloader.preprocess import LaminPreprocessor, additional_postprocess

MAXFILESIZE = 5_000_000_000


def preprocess(adata):
    names = df.columns[1:].tolist()
    adata.obs[names] = df.loc['']
    adata.obs["age_group"] = adata.obs["development_stage_ontology_term_id"]
    adata.obs["is_primary_data"] = True
    adata.obs["cell_culture"] = "True"


preprocessor = Preprocessor(
    is_symbol=False,
    keepdata=True,  # velocyto stuff to keep
    organisms=["NCBITaxon:9606"],
    additional_preprocess=preprocess,
    additional_postprocess=additional_postprocess,
    min_valid_genes_id=6000,
)

file = ln.Artifact.filter(description="/K562_gwps_raw_singlecel").first()
# rpe1_raw_singlecell
# HCT116_filtered_dual_guide orion'
# HEK293T_filtered_dual_guide orion

description = "K562 GWPS Raw Single Cell"
version = '2'
backed = file.open()
if file.size <= MAXFILESIZE:
    adata = backed.to_memory()
    print(adata)
else:
    badata = backed
    print(badata)
try:
    if file.size > MAXFILESIZE:
        print(
            f"dividing the dataset as it is too large: {file.size // 1_000_000_000}Gb"
        )
        num_blocks = int(np.ceil(file.size / (MAXFILESIZE / 2)))
        block_size = int(
            (np.ceil(badata.shape[0] / 30_000) * 30_000) // num_blocks
        )
        print(
            "num blocks ",
            num_blocks,
            "block size ",
            block_size,
            "total elements ",
            badata.shape[0],
        )
        for j in range(num_blocks):
            start_index = j * block_size
            end_index = min((j + 1) * block_size, badata.shape[0])
            block = badata[start_index:end_index]
            block = block.to_memory()
            print(block)
            block = preprocessor(
                block,
                dataset_id=file.stem_uid + "_p" + str(j),
            )
            saved = False
            while not saved:
                try:
                    myfile = ln.Artifact.from_anndata(
                        block,
                        description=description
                        + " p"
                        + str(j)
                        + " ( revises file "
                        + str(file.stem_uid)
                        + " )",
                        version=version,
                    )
                    myfile.save()
                    saved = True
                except OperationalError:
                    print(
                        "Database locked, waiting 30 seconds and retrying..."
                    )
                    time.sleep(10)
            del myfile
            del block
            gc.collect()
    else:
        adata = preprocessor(adata, dataset_id=file.stem_uid)
        saved = False
        while not saved:
            try:
                myfile = ln.Artifact.from_anndata(
                    adata,
                    # revises=file,
                    description=description,
                    version=version,
                )
                myfile.save()
                saved = True
            except OperationalError:
                print(
                    "Database locked, waiting 10 seconds and retrying..."
                )
                time.sleep(10)
        del myfile
        del adata
except ValueError as v:
    if v.args[0].startswith("we cannot work with this organism"):
        print(v)
        continue
    else:
        raise v
except Exception as e:
    if e.args[0].startswith("Dataset dropped due to"):
        print(e)
        continue
    else:
        raise e