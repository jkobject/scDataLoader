# scdataloader

[![codecov](https://codecov.io/gh/jkobject/scDataLoader/branch/main/graph/badge.svg?token=scDataLoader_token_here)](https://codecov.io/gh/jkobject/scDataLoader)
[![CI](https://github.com/jkobject/scDataLoader/actions/workflows/main.yml/badge.svg)](https://github.com/jkobject/scDataLoader/actions/workflows/main.yml)
[![PyPI version](https://badge.fury.io/py/scDataLoader.svg)](https://badge.fury.io/py/scDataLoader)
[![Downloads](https://pepy.tech/badge/scDataLoader)](https://pepy.tech/project/scDataLoader)
[![Downloads](https://pepy.tech/badge/scDataLoader/month)](https://pepy.tech/project/scDataLoader)
[![Downloads](https://pepy.tech/badge/scDataLoader/week)](https://pepy.tech/project/scDataLoader)
[![GitHub issues](https://img.shields.io/github/issues/jkobject/scDataLoader)](https://img.shields.io/github/issues/jkobject/scDataLoader)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://zenodo.org/badge/731248665.svg)](https://doi.org/10.5281/zenodo.10573143)

<img src="scdataloader.png" width="600">

This single cell pytorch dataloader / lighting datamodule is designed to be used with:

- [lamindb](https://lamin.ai/)

and:

- [scanpy](https://scanpy.readthedocs.io/en/stable/)
- [anndata](https://anndata.readthedocs.io/en/latest/)

It allows you to:

1. load thousands of datasets containing millions of cells in a few seconds.
2. preprocess the data per dataset and download it locally (normalization, filtering, etc.)
3. create a more complex single cell dataset
4. extend it to your need

built on top of `lamindb` and the `.mapped()` function by Sergei: https://github.com/Koncopd 

The package has been designed together with the [scPRINT paper](https://doi.org/10.1101/2024.07.29.605556) and [model](https://github.com/cantinilab/scPRINT).

## More

I needed to create this Data Loader for my PhD project. I am using it to load & preprocess thousands of datasets containing millions of cells in a few seconds. I believed that individuals employing AI for single-cell RNA sequencing and other sequencing datasets would eagerly utilize and desire such a tool, which presently does not exist.

![scdataloader.drawio.png](docs/scdataloader.drawio.png)

## Install it from PyPI

```bash
pip install scdataloader
# or
pip install scDataLoader[dev] # for dev dependencies

lamin init --storage ./testdb --name test --schema bionty
```

if you start with lamin and had to do a `lamin init`, you will also need to populate your ontologies. This is because scPRINT is using ontologies to define its cell types, diseases, sexes, ethnicities, etc.

you can do it manually or with our function:

```python
from scdataloader.utils import populate_my_ontology

populate_my_ontology() #to populate everything (recommended) (can take 2-10mns)

populate_my_ontology( #the minimum to the tool
organisms: List[str] = ["NCBITaxon:10090", "NCBITaxon:9606"],
    sex: List[str] = ["PATO:0000384", "PATO:0000383"],
    celltypes = None,
    ethnicities = None,
    assays = None,
    tissues = None,
    diseases = None,
    dev_stages = None,
)
```

### Dev install

If you want to use the latest version of scDataLoader and work on the code yourself use `git clone` and `pip -e` instead of `pip install`.

```bash
git clone https://github.com/jkobject/scDataLoader.git
pip install -e scDataLoader[dev]
```

## Usage

### DataModule usage

```python
# initialize a local lamin database
#! lamin init --storage ./cellxgene --name cellxgene --schema bionty
from scdataloader import utils, Preprocessor, DataModule


# preprocess datasets
preprocessor = Preprocessor(
    do_postp=False,
    force_preprocess=True,
)
adata = preprocessor(adata)

art = ln.Artifact(adata, description="test")
art.save()
ln.Collection(art, name="test", description="test").save()

datamodule = DataModule(
    collection_name="test",
    organisms=["NCBITaxon:9606"], #organism that we will work on
    how="most expr", # for the collator (most expr genes only will be selected)
    max_len=1000, # only the 1000 most expressed
    batch_size=64,
    num_workers=1,
    validation_split=0.1,
)
```

### lightning-free usage (Dataset+Collator+DataLoader)

```python
# initialize a local lamin database
#! lamin init --storage ./cellxgene --name cellxgene --schema bionty

from scdataloader import utils, Preprocessor, SimpleAnnDataset, Collator, DataLoader

# preprocess dataset
preprocessor = Preprocessor(
    do_postp=False,
    force_preprocess=True,
)
adata = preprocessor(adata)

# create dataset
adataset = SimpleAnnDataset(
    adata, obs_to_output=["organism_ontology_term_id"]
)
# create collator
col = Collator(
    organisms="NCBITaxon:9606",
    valid_genes=adata.var_names,
    max_len=2000, #maximum number of genes to use
    how="some" |"most expr"|"random_expr",
    # genelist = [geneA, geneB] if how=='some'
)
# create dataloader
dataloader = DataLoader(
    adataset,
    collate_fn=col,
    batch_size=64,
    num_workers=4,
    shuffle=False,
)

# predict
for batch in tqdm(dataloader):
    gene_pos, expression, depth = (
        batch["genes"],
        batch["x"],
        batch["depth"],
    )
    model.predict(
        gene_pos,
        expression,
        depth,
    )
```

### Usage on all of cellxgene

```python
# initialize a local lamin database
#! lamin init --storage ./cellxgene --name cellxgene --schema bionty

from scdataloader import utils
from scdataloader.preprocess import LaminPreprocessor, additional_postprocess, additional_preprocess

# preprocess datasets
DESCRIPTION='preprocessed by scDataLoader'

cx_dataset = ln.Collection.using(instance="laminlabs/cellxgene").filter(name="cellxgene-census", version='2023-12-15').one()
cx_dataset, len(cx_dataset.artifacts.all())


do_preprocess = LaminPreprocessor(additional_postprocess=additional_postprocess, additional_preprocess=additional_preprocess, skip_validate=True, subset_hvg=0)

preprocessed_dataset = do_preprocess(cx_dataset, name=DESCRIPTION, description=DESCRIPTION, start_at=6, version="2")

# create dataloaders
from scdataloader import DataModule
import tqdm

datamodule = DataModule(
    collection_name="preprocessed dataset",
    organisms=["NCBITaxon:9606"], #organism that we will work on
    how="most expr", # for the collator (most expr genes only will be selected)
    max_len=1000, # only the 1000 most expressed
    batch_size=64,
    num_workers=1,
    validation_split=0.1,
    test_split=0)

for i in tqdm.tqdm(datamodule.train_dataloader()):
    # pass #or do pass
    print(i)
    break

# with lightning:
# Trainer(model, datamodule)

```

see the notebooks in [docs](https://www.jkobject.com/scDataLoader/):

1. [load a dataset](https://www.jkobject.com/scDataLoader/notebooks/1_download_and_preprocess/)
2. [create a dataset](https://www.jkobject.com/scDataLoader/notebooks/2_create_dataloader/)

### command line preprocessing

You can use the command line to preprocess a large database of datasets like here for cellxgene. this allows parallelizing and easier usage.

```bash
scdataloader --instance "laminlabs/cellxgene" --name "cellxgene-census" --version "2023-12-15" --description "preprocessed for scprint" --new_name "scprint main" --start_at 10 >> scdataloader.out
```

### command line usage

The main way to use

> please refer to the [scPRINT documentation](https://www.jkobject.com/scPRINT/) and [lightning documentation](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_intermediate.html) for more information on command line usage

## FAQ

### how to update my ontologies?

```bash
import bionty as bt
bt.reset_sources()

# Run via CLI: lamin load <your instance>

import lnschema_bionty as lb
lb.dev.sync_bionty_source_to_latest()
```

### how to load all ontologies?

```python
from scdataloader import utils
utils.populate_ontologies() # this might take from 5-20mins
```

## Development

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [lamin.ai](https://lamin.ai/)
- [scanpy](https://scanpy.readthedocs.io/en/stable/)
- [anndata](https://anndata.readthedocs.io/en/latest/)
- [scprint](https://www.jkobject.com/scPRINT/)

Awesome single cell dataloader created by @jkobject
