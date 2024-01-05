import numpy as np
from torch.utils.data import DataLoader as TorchLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import WeightedRandomSampler
from scdataloader.mapped import MappedDataset
from typing import Union
import torch

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


class DataLoader(TorchLoader):
    """
    Base class for all data loaders
    """

    def __init__(
        self,
        mapped_dataset: MappedDataset,
        batch_size: int = 32,
        weight_scaler: int = 30,
        label_to_weight: list = [],
        validation_split: float = 0.2,
        num_workers: int = 4,
        collate_fn=default_collate,
        sampler=None,
        **kwargs,
    ):
        self.validation_split = validation_split
        self.dataset = mapped_dataset

        self.batch_idx = 0
        self.batch_size = batch_size
        self.n_samples = len(self.dataset)
        if sampler is None:
            self.sampler, self.valid_sampler = self._split_sampler(
                self.validation_split,
                weight_scaler=weight_scaler,
                label_to_weight=label_to_weight,
            )
        else:
            self.sampler = sampler
            self.valid_sampler = None

        self.init_kwargs = {
            "dataset": self.dataset,
            "batch_size": batch_size,
            "collate_fn": collate_fn,
            "num_workers": num_workers,
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs, **kwargs)

    def _split_sampler(self, split, label_to_weight=[], weight_scaler: int = 30):
        idx_full = np.arange(self.n_samples)
        np.random.shuffle(idx_full)
        if len(label_to_weight) > 0:
            weights = self.dataset.get_label_weights(
                label_to_weight, scaler=weight_scaler
            )
        else:
            weights = np.ones(self.n_samples)
        if isinstance(split, int):
            assert (
                split < self.n_samples
            ), "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)
        if len_valid == 0:
            self.train_idx = idx_full
        else:
            self.valid_idx = idx_full[0:len_valid]
            self.train_idx = np.delete(idx_full, np.arange(0, len_valid))
            valid_weights = weights.copy()
            valid_weights[self.train_idx] = 0
            # TODO: should we do weighted random sampling for validation set?
            valid_sampler = WeightedRandomSampler(
                valid_weights, len_valid, replacement=True
            )
        train_weights = weights.copy()
        train_weights[self.valid_idx] = 0
        train_sampler = WeightedRandomSampler(
            train_weights, len(self.train_idx), replacement=True
        )
        # turn off shuffle option which is mutually exclusive with sampler

        return (
            (train_sampler, valid_sampler) if len_valid != 0 else (train_sampler, None)
        )

    def get_valid_dataloader(self):
        if self.valid_sampler is None:
            raise ValueError("No validation set is configured.")
        return DataLoader(
            self.dataset, batch_size=self.batch_size, sampler=self.valid_sampler
        )


def weighted_random_mask_value(
    values: Union[torch.Tensor, np.ndarray],
    mask_ratio: float = 0.15,
    mask_value: int = -1,
    important_elements: Union[torch.Tensor, np.ndarray] = np.array([]),
    important_weight: int = 0,
    pad_value: int = 0,
) -> torch.Tensor:
    """
    Randomly mask a batch of data.

    Args:
        values (array-like):
            A batch of tokenized data, with shape (batch_size, n_features).
        mask_ratio (float): The ratio of genes to mask, default to 0.15.
        mask_value (int): The value to mask with, default to -1.
        pad_value (int): The value of padding in the values, will be kept unchanged.

    Returns:
        torch.Tensor: A tensor of masked data.
    """
    if isinstance(values, torch.Tensor):
        # it is crutial to clone the tensor, otherwise it changes the original tensor
        values = values.clone().detach().numpy()
    else:
        values = values.copy()

    for i in range(len(values)):
        row = values[i]
        non_padding_idx = np.nonzero(row - pad_value)[0]
        non_padding_idx = np.setdiff1d(non_padding_idx, do_not_pad_index)
        n_mask = int(len(non_padding_idx) * mask_ratio)
        mask_idx = np.random.choice(non_padding_idx, n_mask, replace=False)
        row[mask_idx] = mask_value
    return torch.from_numpy(values).float()
