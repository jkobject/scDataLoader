import numpy as np
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import WeightedRandomSampler
from scdataloader.mapped import MappedDataset
from torch.utils import data
from typing import Union
import torch
import lightning as L


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
        mapped_dataset: MappedDataset,
        weight_scaler: int = 30,
        label_to_weight: list = [],
        validation_split: float = 0.2,
        test_split: float = 0,
        **kwargs,
    ):
        self.validation_split = validation_split
        self.test_split = test_split
        self.dataset = mapped_dataset
        self.kwargs = kwargs
        self.n_samples = len(self.dataset)
        self.weight_scaler = weight_scaler
        self.label_to_weight = label_to_weight
        super().__init__()

    def setup(self, stage=None):
        idx_full = np.arange(self.n_samples)
        np.random.shuffle(idx_full)
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
        if len_valid > 0:
            valid_idx = idx_full[0:len_valid]
            valid_weights = weights.copy()
            valid_weights[~valid_idx] = 0
            self.valid_sampler = WeightedRandomSampler(
                valid_weights, len_valid, replacement=True
            )
        else:
            self.valid_sampler = None
        if len_test > 0:
            test_idx = idx_full[len_valid : len_valid + len_test]
            test_weights = weights.copy()
            test_weights[~test_idx] = 0
            self.test_sampler = WeightedRandomSampler(
                valid_weights, len_valid, replacement=True
            )
        else:
            self.test_sampler = None

        train_idx = idx_full[len_valid + len_test :]
        train_weights = weights.copy()
        train_weights[~train_idx] = 0
        self.train_sampler = WeightedRandomSampler(
            train_weights, len(train_idx), replacement=True
        )

    def train_dataloader(self):
        return data.DataLoader(self.dataset, sampler=self.train_sampler, **self.kwargs)

    def val_dataloader(self):
        return (
            data.DataLoader(self.dataset, sampler=self.val_sampler, **self.kwargs)
            if self.val_sampler is not None
            else None
        )

    def test_dataloader(self):
        return (
            data.DataLoader(self.dataset, sampler=self.test_sampler, **self.kwargs)
            if self.test_sampler is not None
            else None
        )

    # def teardown(self):
    # clean up state after the trainer stops, delete files...
    # called on every process in DDP
    # pass
