import numpy as np
import pandas as pd
import lamindb as ln

from torch.utils.data.sampler import (
    WeightedRandomSampler,
    SubsetRandomSampler,
    SequentialSampler,
)
from torch.utils.data import DataLoader
import lightning as L

from typing import Optional

from .data import Dataset
from .collator import Collator
from .mapped import MappedDataset
from .utils import getBiomartTable

# TODO: put in config
COARSE_TISSUE = {
    "adipose tissue": "",
    "bladder organ": "",
    "blood": "",
    "bone marrow": "",
    "brain": "",
    "breast": "",
    "esophagus": "",
    "eye": "",
    "embryo": "",
    "fallopian tube": "",
    "gall bladder": "",
    "heart": "",
    "intestine": "",
    "kidney": "",
    "liver": "",
    "lung": "",
    "lymph node": "",
    "musculature of body": "",
    "nose": "",
    "ovary": "",
    "pancreas": "",
    "placenta": "",
    "skin of body": "",
    "spinal cord": "",
    "spleen": "",
    "stomach": "",
    "thymus": "",
    "thyroid gland": "",
    "tongue": "",
    "uterus": "",
}

COARSE_ANCESTRY = {
    "African": "",
    "Chinese": "",
    "East Asian": "",
    "Eskimo": "",
    "European": "",
    "Greater Middle Eastern  (Middle Eastern, North African or Persian)": "",
    "Hispanic or Latin American": "",
    "Native American": "",
    "Oceanian": "",
    "South Asian": "",
}

COARSE_DEVELOPMENT_STAGE = {
    "Embryonic human": "",
    "Fetal": "",
    "Immature": "",
    "Mature": "",
}

COARSE_ASSAY = {
    "10x 3'": "",
    "10x 5'": "",
    "10x multiome": "",
    "CEL-seq2": "",
    "Drop-seq": "",
    "GEXSCOPE technology": "",
    "inDrop": "",
    "microwell-seq": "",
    "sci-Plex": "",
    "sci-RNA-seq": "",
    "Seq-Well": "",
    "Slide-seq": "",
    "Smart-seq": "",
    "SPLiT-seq": "",
    "TruDrop": "",
    "Visium Spatial Gene Expression": "",
}


class DataModule(L.LightningDataModule):
    """
    Base class for all data loaders
    """

    def __init__(
        self,
        mdataset: Optional[MappedDataset] = None,
        collection_name=None,
        organisms: list = ["NCBITaxon:9606"],
        weight_scaler: int = 30,
        train_oversampling=1,
        label_to_weight: list = [],
        label_to_pred: list = [],
        validation_split: float = 0.2,
        test_split: float = 0,
        use_default_col=True,
        all_labels=[],
        hierarchical_labels=[],
        how="most expr",
        organism_name="organism_ontology_term_id",
        max_len=1000,
        add_zero_genes=100,
        do_gene_pos=True,
        gene_embeddings="",
        gene_position_tolerance=10_000,
        **kwargs,
    ):
        """
        Initializes the DataModule.

        Args:
            dataset (MappedDataset): The dataset to be used.
            weight_scaler (int, optional): The weight scaler for weighted random sampling. Defaults to 30.
            label_to_weight (list, optional): List of labels to weight. Defaults to [].
            validation_split (float, optional): The proportion of the dataset to include in the validation split. Defaults to 0.2.
            test_split (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.
            **kwargs: Additional keyword arguments passed to the pytorch DataLoader.
        """
        if collection_name is not None:
            mdataset = Dataset(
                ln.Collection.filter(name=collection_name).first(),
                organisms=organisms,
                obs=all_labels,
                clss_to_pred=label_to_pred,
                hierarchical_clss=hierarchical_labels,
            )
            print(mdataset)
        # and location
        if do_gene_pos:
            # and annotations
            biomart = getBiomartTable(
                attributes=["start_position", "chromosome_name"]
            ).set_index("ensembl_gene_id")
            biomart = biomart.loc[~biomart.index.duplicated(keep="first")]
            biomart = biomart.sort_values(by=["chromosome_name", "start_position"])
            c = []
            i = 0
            prev_position = -100000
            prev_chromosome = None
            for _, r in biomart.iterrows():
                if (
                    r["chromosome_name"] != prev_chromosome
                    or r["start_position"] - prev_position > gene_position_tolerance
                ):
                    i += 1
                c.append(i)
                prev_position = r["start_position"]
                prev_chromosome = r["chromosome_name"]
            print(f"reduced the size to {len(set(c))/len(biomart)}")
            biomart["pos"] = c
            mdataset.genedf = biomart.loc[
                mdataset.genedf[mdataset.genedf.index.isin(biomart.index)].index
            ]
            self.gene_pos = mdataset.genedf["pos"].tolist()

        if gene_embeddings != "":
            mdataset.genedf = mdataset.genedf.join(
                pd.read_parquet(gene_embeddings), how="inner"
            )
            if do_gene_pos:
                self.gene_pos = mdataset.genedf["pos"].tolist()
        self.labels = {k: len(v) for k, v in mdataset.class_topred.items()}
        # we might want not to order the genes by expression (or do it?)
        # we might want to not introduce zeros and
        if use_default_col:
            kwargs["collate_fn"] = Collator(
                organisms=organisms,
                how=how,
                valid_genes=mdataset.genedf.index.tolist(),
                max_len=max_len,
                add_zero_genes=add_zero_genes,
                org_to_id=mdataset.encoder[organism_name],
                tp_name="heat_diff",
                organism_name=organism_name,
                class_names=label_to_weight,
            )
        self.validation_split = validation_split
        self.test_split = test_split
        self.dataset = mdataset
        self.kwargs = kwargs
        self.n_samples = len(mdataset)
        self.weight_scaler = weight_scaler
        self.train_oversampling = train_oversampling
        self.label_to_weight = label_to_weight
        super().__init__()

    @property
    def decoders(self):
        decoders = {}
        for k, v in self.dataset.encoder.items():
            decoders[k] = {va: ke for ke, va in v.items()}
        return decoders

    @property
    def cls_hierarchy(self):
        cls_hierarchy = {}
        for k, dic in self.dataset.class_groupings.items():
            rdic = {}
            for sk, v in dic.items():
                rdic[self.dataset.encoder[k][sk]] = [
                    self.dataset.encoder[k][i] for i in list(v)
                ]
            cls_hierarchy[k] = rdic
        return cls_hierarchy

    @property
    def genes(self):
        return self.dataset.genedf.index.tolist()

    def setup(self, stage=None):
        """
        setup method is used to prepare the data for the training, validation, and test sets.
        It shuffles the data, calculates weights for each set, and creates samplers for each set.

        Args:
            stage (str, optional): The stage of the model training process.
            It can be either 'fit' or 'test'. Defaults to None.
        """

        if len(self.label_to_weight) > 0:
            weights = self.dataset.get_label_weights(
                self.label_to_weight, scaler=self.weight_scaler
            )
        else:
            weights = np.ones(self.n_samples)
        if isinstance(self.validation_split, int):
            len_valid = self.validation_split
        else:
            len_valid = int(self.n_samples * self.validation_split)
        if isinstance(self.test_split, int):
            len_test = self.test_split
        else:
            len_test = int(self.n_samples * self.test_split)
        assert (
            len_test + len_valid < self.n_samples
        ), "test set + valid set size is configured to be larger than entire dataset."
        idx_full = np.arange(self.n_samples)
        if len_test > 0:
            # this way we work on some never seen datasets
            # keeping at least one
            len_test = (
                len_test
                if len_test > self.dataset.mapped_dataset.n_obs_list[0]
                else self.dataset.mapped_dataset.n_obs_list[0]
            )
            cs = 0
            test_datasets = []
            print("these files will be considered test datasets:")
            for i, c in enumerate(self.dataset.mapped_dataset.n_obs_list):
                if cs + c > len_test:
                    break
                else:
                    print("    " + self.dataset.mapped_dataset.path_list[i].path)
                    test_datasets.append(self.dataset.mapped_dataset.path_list[i].path)
                    cs += c

            len_test = cs
            print("perc test: ", len_test / self.n_samples)
            test_idx = idx_full[:len_test]
            idx_full = idx_full[len_test:]
            self.test_sampler = SequentialSampler(test_idx)
        else:
            self.test_sampler = None
            test_datasets = None

        np.random.shuffle(idx_full)
        if len_valid > 0:
            valid_idx = idx_full[:len_valid]
            idx_full = idx_full[len_valid:]
            self.valid_sampler = SubsetRandomSampler(valid_idx)
        else:
            self.valid_sampler = None

        weights[~idx_full] = 0
        self.train_sampler = WeightedRandomSampler(
            weights,
            int(len(idx_full) * self.train_oversampling),
            replacement=True,
        )
        return test_datasets

    def train_dataloader(self, **kwargs):
        return DataLoader(
            self.dataset, sampler=self.train_sampler, **self.kwargs, **kwargs
        )

    def val_dataloader(self):
        return (
            DataLoader(self.dataset, sampler=self.valid_sampler, **self.kwargs)
            if self.valid_sampler is not None
            else None
        )

    def test_dataloader(self):
        return (
            DataLoader(self.dataset, sampler=self.test_sampler, **self.kwargs)
            if self.test_sampler is not None
            else None
        )

    # def teardown(self):
    # clean up state after the trainer stops, delete files...
    # called on every process in DDP
    # pass
