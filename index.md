# scdataloader

[![codecov](https://codecov.io/gh/jkobject/scDataLoader/branch/main/graph/badge.svg?token=scDataLoader_token_here)](https://codecov.io/gh/jkobject/scDataLoader)
[![CI](https://github.com/jkobject/scDataLoader/actions/workflows/main.yml/badge.svg)](https://github.com/jkobject/scDataLoader/actions/workflows/main.yml)
[![PyPI version](https://badge.fury.io/py/scDataLoader.svg)](https://badge.fury.io/py/scDataLoader)
[![Documentation Status](https://readthedocs.org/projects/scDataLoader/badge/?version=latest)](https://scDataLoader.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://pepy.tech/badge/scDataLoader)](https://pepy.tech/project/scDataLoader)
[![Downloads](https://pepy.tech/badge/scDataLoader/month)](https://pepy.tech/project/scDataLoader)
[![Downloads](https://pepy.tech/badge/scDataLoader/week)](https://pepy.tech/project/scDataLoader)
[![GitHub issues](https://img.shields.io/github/issues/jkobject/scDataLoader)](https://img.shields.io/github/issues/jkobject/scDataLoader)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://img.shields.io/badge/DOI-10.1101%2F2024.07.29.605556-blue)](https://doi.org/10.1101/2024.07.29.605556)

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

built on top of `lamindb` and the `.mapped()` function by Sergey: https://github.com/Koncopd 

## More

I needed to create this Data Loader for my PhD project. I am using it to load & preprocess thousands of datasets containing millions of cells in a few seconds. I believed that individuals employing AI for single-cell RNA sequencing and other sequencing datasets would eagerly utilize and desire such a tool, which presently does not exist.

![scdataloader.drawio.png](scdataloader.drawio.png)

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

### Direct Usage

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

### command line preprocessing

You can use the command line to preprocess a large database of datasets like here for cellxgene. this allows parallelizing and easier usage.

```bash
scdataloader --instance "laminlabs/cellxgene" --name "cellxgene-census" --version "2023-12-15" --description "preprocessed for scprint" --new_name "scprint main" --start_at 10 >> scdataloader.out
```

### command line usage

The main way to use

> please refer to the [scPRINT documentation](https://www.jkobject.com/scPRINT/) and [lightning documentation](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_intermediate.html) for more information on command line usage

## Development

Read the [CONTRIBUTING.md](../CONTRIBUTING.md) file.

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## Acknowledgments

- [lamin.ai](https://lamin.ai/)
- [scanpy](https://scanpy.readthedocs.io/en/stable/)
- [anndata](https://anndata.readthedocs.io/en/latest/)
- [scprint](https://www.jkobject.com/scPRINT/)

Awesome single cell dataloader created by @jkobject