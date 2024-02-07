import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler, SubsetRandomSampler
from scdataloader.mapped import MappedDataset
from torch.utils.data import DataLoader
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
        dataset: MappedDataset,
        weight_scaler: int = 30,
        label_to_weight: list = [],
        validation_split: float = 0.2,
        test_split: float = 0,
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
        self.validation_split = validation_split
        self.test_split = test_split
        self.dataset = dataset
        self.kwargs = kwargs
        self.n_samples = len(self.dataset)
        self.weight_scaler = weight_scaler
        self.label_to_weight = label_to_weight
        super().__init__()

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
            print("these files will be considered test datasets")
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
            # test_weights = weights.copy()
            # test_weights[~test_idx] = 0
            self.test_sampler = SubsetRandomSampler(test_idx)
        else:
            self.test_sampler = None
            test_datasets = None

        if len_valid > 0:
            valid_idx = idx_full[:len_valid]
            idx_full = idx_full[len_valid:]
            # valid_weights = weights.copy()
            # valid_weights[~valid_idx] = 0
            self.valid_sampler = SubsetRandomSampler(valid_idx)
        else:
            self.valid_sampler = None

        np.random.shuffle(idx_full)
        train_idx = idx_full[len_valid:]
        train_weights = weights.copy()
        train_weights[~train_idx] = 0
        self.train_sampler = WeightedRandomSampler(
            train_weights, len(train_idx), replacement=True
        )
        return test_datasets

    def train_dataloader(self):
        return DataLoader(self.dataset, sampler=self.train_sampler, **self.kwargs)

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
