# scdataloader

[![codecov](https://codecov.io/gh/jkobject/scDataLoader/branch/main/graph/badge.svg?token=scDataLoader_token_here)](https://codecov.io/gh/jkobject/scDataLoader)
[![CI](https://github.com/jkobject/scDataLoader/actions/workflows/main.yml/badge.svg)](https://github.com/jkobject/scDataLoader/actions/workflows/main.yml)

Awesome single cell dataloader created by @jkobject

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

## More

I needed to create this Data Loader for my PhD project. I am using it to load & preprocess thousands of datasets containing millions of cells in a few seconds. I believed that individuals employing AI for single-cell RNA sequencing and other sequencing datasets would eagerly utilize and desire such a tool, which presently does not exist.

![scdataloader.drawio.png](scdataloader.drawio.png)

## Install it from PyPI

```bash
pip install scdataloader
```

## Usage

see the notebooks in [docs](https://jkobject.github.io/scDataLoader/):

1. [load a dataset](https://jkobject.github.io/scDataLoader/notebooks/01_load_dataset.html)
2. [create a dataset](https://jkobject.github.io/scDataLoader/notebooks/02_create_dataset.html)

## Development

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.
