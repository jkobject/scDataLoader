# scdataloader

[![codecov](https://codecov.io/gh/jkobject/scDataLoader/branch/main/graph/badge.svg?token=scDataLoader_token_here)](https://codecov.io/gh/jkobject/scDataLoader)
[![CI](https://github.com/jkobject/scDataLoader/actions/workflows/main.yml/badge.svg)](https://github.com/jkobject/scDataLoader/actions/workflows/main.yml)
[![DOI](https://zenodo.org/badge/731248665.svg)](https://zenodo.org/doi/10.5281/zenodo.10573143)


Awesome single cell dataloader created by @jkobject 

built on top of `lamindb` and the `.mapped()` function by Sergey: https://github.com/Koncopd 

This data loader is designed to be used with:

- [lamindb](https://lamin.ai/)

and:

- [scanpy](https://scanpy.readthedocs.io/en/stable/)
- [anndata](https://anndata.readthedocs.io/en/latest/)

It allows you to:

1. load thousands of datasets containing millions of cells in a few seconds.
2. preprocess the data per dataset and download it locally (normalization, filtering, etc.)
3. create a more complex single cell dataset
4. extend it to your need

## About

the idea is to use it to train models like scGPT / GeneFormer (and soon, scPrint ;)). It is: 

1. loading from lamin 
2. doing some dataset specific preprocessing if needed 
3. creating a dataset object on top of .mapped() (that is needed for mapping genes, cell labels etc..)
4. passing it to a dataloader object that can work with it correctly

Currently one would have to use the preprocess function to make the dataset fit for different tools like scGPT / Geneformer. But I would want to enable it through different Collators. This is still missing and a WIP... (please do contribute!)

![docs/scdataloader.drawio.png](docs/scdataloader.drawio.png)

## Install it from PyPI

```bash
pip install scdataloader
```

### Install it locally and run the notebooks:

```bash
git clone https://github.com/jkobject/scDataLoader.git
cd scDataLoader
poetry install
```
then run the notebooks with the poetry installed environment

## Usage

```python
# initialize a local lamin database
# !lamin init --storage ~/scdataloader --schema bionty

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

see the notebooks in [docs](https://jkobject.github.io/scDataLoader/):

1. [load a dataset](https://jkobject.github.io/scDataLoader/notebooks/01_load_dataset.html)
2. [create a dataset](https://jkobject.github.io/scDataLoader/notebooks/02_create_dataset.html)

## Development

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.
