import os
from typing import Optional, Sequence, Union

import lamindb as ln
import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.sampler import (
    RandomSampler,
    SequentialSampler,
    SubsetRandomSampler,
    WeightedRandomSampler,
)

from .collator import Collator
from .data import Dataset
from .utils import getBiomartTable, slurm_restart_count

FILE_DIR = os.path.dirname(os.path.abspath(__file__))


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        collection_name: str,
        clss_to_weight: list = ["organism_ontology_term_id"],
        organisms: list = ["NCBITaxon:9606"],
        weight_scaler: int = 10,
        train_oversampling_per_epoch: float = 0.1,
        validation_split: float = 0.2,
        test_split: float = 0,
        gene_embeddings: str = "",
        use_default_col: bool = True,
        gene_position_tolerance: int = 10_000,
        # this is for the mappedCollection
        clss_to_predict: list = ["organism_ontology_term_id"],
        hierarchical_clss: list = [],
        # this is for the collator
        how: str = "random expr",
        organism_name: str = "organism_ontology_term_id",
        max_len: int = 1000,
        add_zero_genes: int = 100,
        replacement: bool = True,
        do_gene_pos: Union[bool, str] = True,
        tp_name: Optional[str] = None,  # "heat_diff"
        assays_to_drop: list = [
            # "EFO:0008853", #patch seq
            # "EFO:0010961", # visium
            "EFO:0030007",  # ATACseq
            # "EFO:0030062", # slide-seq
        ],
        metacell_mode: float = 0.0,
        get_knn_cells: bool = False,
        modify_seed_on_requeue: bool = True,
        **kwargs,
    ):
        """
        DataModule a pytorch lighting datamodule directly from a lamin Collection.
        it can work with bare pytorch too

        It implements train / val / test dataloaders. the train is weighted random, val is random, test is one to many separated datasets.
        This is where the mappedCollection, dataset, and collator are combined to create the dataloaders.

        Args:
            collection_name (str): The lamindb collection to be used.
            organisms (list, optional): The organisms to include in the dataset. Defaults to ["NCBITaxon:9606"].
            weight_scaler (int, optional): how much more you will see the most present vs less present category.
            train_oversampling_per_epoch (float, optional): The proportion of the dataset to include in the training set for each epoch. Defaults to 0.1.
            validation_split (float, optional): The proportion of the dataset to include in the validation split. Defaults to 0.2.
            test_split (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.
                it will use a full dataset and will round to the nearest dataset's cell count.
            gene_embeddings (str, optional): The path to the gene embeddings file. Defaults to "".
                the file must have ensembl_gene_id as index.
                This is used to subset the available genes further to the ones that have embeddings in your model.
            use_default_col (bool, optional): Whether to use the default collator. Defaults to True.
            gene_position_tolerance (int, optional): The tolerance for gene position. Defaults to 10_000.
                any genes within this distance of each other will be considered at the same position.
            clss_to_weight (list, optional): List of labels to weight in the trainer's weighted random sampler. Defaults to [].
            assays_to_drop (list, optional): List of assays to drop from the dataset. Defaults to [].
            do_gene_pos (Union[bool, str], optional): Whether to use gene positions. Defaults to True.
            max_len (int, optional): The maximum length of the input tensor. Defaults to 1000.
            add_zero_genes (int, optional): The number of zero genes to add to the input tensor. Defaults to 100.
            how (str, optional): The method to use for the collator. Defaults to "random expr".
            organism_name (str, optional): The name of the organism. Defaults to "organism_ontology_term_id".
            tp_name (Optional[str], optional): The name of the timepoint. Defaults to None.
            hierarchical_clss (list, optional): List of hierarchical classes. Defaults to [].
            metacell_mode (float, optional): The probability of using metacell mode. Defaults to 0.0.
            clss_to_predict (list, optional): List of classes to predict. Defaults to ["organism_ontology_term_id"].
            modify_seed_on_requeue (bool, optional): Whether to modify the seed on requeue. Defaults to True.
            get_knn_cells (bool, optional): Whether to get the k-nearest neighbors of each queried cells. Defaults to False.
            **kwargs: Additional keyword arguments passed to the pytorch DataLoader.
            see @file data.py and @file collator.py for more details about some of the parameters
        """
        if collection_name is not None:
            mdataset = Dataset(
                ln.Collection.filter(name=collection_name).first(),
                organisms=organisms,
                clss_to_predict=clss_to_predict,
                hierarchical_clss=hierarchical_clss,
                metacell_mode=metacell_mode,
                get_knn_cells=get_knn_cells,
            )
        # and location
        self.metacell_mode = bool(metacell_mode)
        self.gene_pos = None
        self.collection_name = collection_name
        if do_gene_pos:
            if type(do_gene_pos) is str:
                print("seeing a string: loading gene positions as biomart parquet file")
                biomart = pd.read_parquet(do_gene_pos)
            else:
                # and annotations
                if organisms != ["NCBITaxon:9606"]:
                    raise ValueError(
                        "need to provide your own table as this automated function only works for humans for now"
                    )
                biomart = getBiomartTable(
                    attributes=["start_position", "chromosome_name"],
                    useCache=True,
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
                print(f"reduced the size to {len(set(c)) / len(biomart)}")
                biomart["pos"] = c
            mdataset.genedf = mdataset.genedf.join(biomart, how="inner")
            self.gene_pos = mdataset.genedf["pos"].astype(int).tolist()

        if gene_embeddings != "":
            mdataset.genedf = mdataset.genedf.join(
                pd.read_parquet(gene_embeddings), how="inner"
            )
            if do_gene_pos:
                self.gene_pos = mdataset.genedf["pos"].tolist()
        self.classes = {k: len(v) for k, v in mdataset.class_topred.items()}
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
                tp_name=tp_name,
                organism_name=organism_name,
                class_names=clss_to_predict,
            )
        self.validation_split = validation_split
        self.test_split = test_split
        self.dataset = mdataset
        self.replacement = replacement
        self.kwargs = kwargs
        if "sampler" in self.kwargs:
            self.kwargs.pop("sampler")
        self.assays_to_drop = assays_to_drop
        self.n_samples = len(mdataset)
        self.weight_scaler = weight_scaler
        self.train_oversampling_per_epoch = train_oversampling_per_epoch
        self.clss_to_weight = clss_to_weight
        self.train_weights = None
        self.train_labels = None
        self.modify_seed_on_requeue = modify_seed_on_requeue
        self.nnz = None
        self.restart_num = 0
        self.test_datasets = []
        self.test_idx = []
        super().__init__()

    def __repr__(self):
        return (
            f"DataLoader(\n"
            f"\twith a dataset=({self.dataset.__repr__()}\n)\n"
            f"\tvalidation_split={self.validation_split},\n"
            f"\ttest_split={self.test_split},\n"
            f"\tn_samples={self.n_samples},\n"
            f"\tweight_scaler={self.weight_scaler},\n"
            f"\ttrain_oversampling_per_epoch={self.train_oversampling_per_epoch},\n"
            f"\tassays_to_drop={self.assays_to_drop},\n"
            f"\tnum_datasets={len(self.dataset.mapped_dataset.storages)},\n"
            f"\ttest datasets={str(self.test_datasets)},\n"
            f"perc test: {str(len(self.test_idx) / self.n_samples)},\n"
            f"\tclss_to_weight={self.clss_to_weight}\n"
            + ("\twith train_dataset size of=(" + str(len(self.idx_full)) + ")\n)")
            if self.idx_full is not None
            else ")"
        )

    @property
    def decoders(self):
        """
        decoders the decoders for any labels that would have been encoded

        Returns:
            dict[str, dict[int, str]]
        """
        decoders = {}
        for k, v in self.dataset.encoder.items():
            decoders[k] = {va: ke for ke, va in v.items()}
        return decoders

    @property
    def labels_hierarchy(self):
        """
        labels_hierarchy the hierarchy of labels for any cls that would have a hierarchy

        Returns:
            dict[str, dict[str, str]]
        """
        labels_hierarchy = {}
        for k, dic in self.dataset.labels_groupings.items():
            rdic = {}
            for sk, v in dic.items():
                rdic[self.dataset.encoder[k][sk]] = [
                    self.dataset.encoder[k][i] for i in list(v)
                ]
            labels_hierarchy[k] = rdic
        return labels_hierarchy

    @property
    def genes(self):
        """
        genes the genes used in this datamodule

        Returns:
            list
        """
        return self.dataset.genedf.index.tolist()

    @property
    def num_datasets(self):
        return len(self.dataset.mapped_dataset.storages)

    def setup(self, stage=None):
        """
        setup method is used to prepare the data for the training, validation, and test sets.
        It shuffles the data, calculates weights for each set, and creates samplers for each set.

        Args:
            stage (str, optional): The stage of the model training process.
            It can be either 'fit' or 'test'. Defaults to None.
        """
        SCALE = 10
        if "nnz" in self.clss_to_weight and self.weight_scaler > 0:
            self.nnz = self.dataset.mapped_dataset.get_merged_labels("nnz")
            self.clss_to_weight.remove("nnz")
            (
                (self.nnz.max() / SCALE)
                / ((1 + self.nnz - self.nnz.min()) + (self.nnz.max() / SCALE))
            ).min()
        if len(self.clss_to_weight) > 0 and self.weight_scaler > 0:
            weights, labels = self.dataset.get_label_weights(
                self.clss_to_weight,
                scaler=self.weight_scaler,
                return_categories=True,
            )
        else:
            weights = np.ones(1)
            labels = np.zeros(self.n_samples, dtype=int)
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

        idx_full = []
        if len(self.assays_to_drop) > 0:
            badloc = np.isin(
                self.dataset.mapped_dataset.get_merged_labels("assay_ontology_term_id"),
                self.assays_to_drop,
            )
            idx_full = np.arange(len(labels))[~badloc]
        else:
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
            for i, c in enumerate(self.dataset.mapped_dataset.n_obs_list):
                if cs + c > len_test:
                    break
                else:
                    self.test_datasets.append(
                        self.dataset.mapped_dataset.path_list[i].path
                    )
                    cs += c
            len_test = cs
            self.test_idx = idx_full[:len_test]
            idx_full = idx_full[len_test:]
        else:
            self.test_idx = None

        np.random.shuffle(idx_full)
        if len_valid > 0:
            self.valid_idx = idx_full[:len_valid].copy()
            # store it for later
            idx_full = idx_full[len_valid:]
        else:
            self.valid_idx = None
        weights = np.concatenate([weights, np.zeros(1)])
        labels[~np.isin(np.arange(self.n_samples), idx_full)] = len(weights) - 1
        # some labels will now not exist anymore as replaced by len(weights) - 1.
        # this means that the associated weights should be 0.
        # by doing np.bincount(labels)*weights this will be taken into account
        self.train_weights = weights
        self.train_labels = labels
        self.idx_full = idx_full
        return self.test_datasets

    def train_dataloader(self, **kwargs):
        # train_sampler = WeightedRandomSampler(
        #    self.train_weights[self.train_labels],
        #    int(self.n_samples*self.train_oversampling_per_epoch),
        #    replacement=True,
        # )
        try:
            train_sampler = LabelWeightedSampler(
                label_weights=self.train_weights,
                labels=self.train_labels,
                num_samples=int(self.n_samples * self.train_oversampling_per_epoch),
                element_weights=self.nnz,
                replacement=self.replacement,
                restart_num=self.restart_num,
                modify_seed_on_requeue=self.modify_seed_on_requeue,
            )
        except ValueError as e:
            raise ValueError(e + "have you run `datamodule.setup()`?")
        return DataLoader(
            self.dataset,
            sampler=train_sampler,
            **self.kwargs,
            **kwargs,
        )

    def val_dataloader(self):
        return (
            DataLoader(
                self.dataset,
                sampler=SubsetRandomSampler(self.valid_idx),
                **self.kwargs,
            )
            if self.valid_idx is not None
            else None
        )

    def test_dataloader(self):
        return (
            DataLoader(
                self.dataset, sampler=SequentialSampler(self.test_idx), **self.kwargs
            )
            if self.test_idx is not None
            else None
        )

    def predict_dataloader(self):
        return DataLoader(
            self.dataset,
            sampler=SubsetRandomSampler(self.idx_full),
            **self.kwargs,
        )

    # def teardown(self):
    # clean up state after the trainer stops, delete files...
    # called on every process in DDP
    # pass


class LabelWeightedSampler(Sampler[int]):
    label_weights: Sequence[float]
    klass_indices: Sequence[Sequence[int]]
    num_samples: int
    nnz: Optional[Sequence[int]]
    replacement: bool
    restart_num: int
    modify_seed_on_requeue: bool
    # when we use, just set weights for each classes(here is: np.ones(num_classes)), and labels of a dataset.
    # this will result a class-balanced sampling, no matter how imbalance the labels are.

    def __init__(
        self,
        label_weights: Sequence[float],
        labels: Sequence[int],
        num_samples: int,
        replacement: bool = True,
        element_weights: Sequence[float] = None,
        restart_num: int = 0,
        modify_seed_on_requeue: bool = True,
    ) -> None:
        """

        :param label_weights: list(len=num_classes)[float], weights for each class.
        :param labels: list(len=dataset_len)[int], labels of a dataset.
        :param num_samples: number of samples.
        :param restart_num: if we are continuing a previous run, we need to restart the sampler from the same point.
        """

        super(LabelWeightedSampler, self).__init__(None)
        # reweight labels from counter otherwsie same weight to labels that have many elements vs a few
        label_weights = np.array(label_weights) * np.bincount(labels)

        self.label_weights = torch.as_tensor(label_weights, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.int)
        self.element_weights = (
            torch.as_tensor(element_weights, dtype=torch.float32)
            if element_weights is not None
            else None
        )
        self.replacement = replacement
        self.num_samples = num_samples
        self.restart_num = slurm_restart_count(use_mine=True) + restart_num
        self.modify_seed_on_requeue = modify_seed_on_requeue
        # list of tensor.
        self.klass_indices = [
            (self.labels == i_klass).nonzero().squeeze(1)
            for i_klass in range(len(label_weights))
        ]
        self.klass_sizes = [len(klass_indices) for klass_indices in self.klass_indices]

    def __iter__(self):
        sample_labels = torch.multinomial(
            self.label_weights,
            num_samples=self.num_samples,
            replacement=True,
            generator=None
            if self.restart_num == 0 and not self.modify_seed_on_requeue
            else torch.Generator().manual_seed(self.restart_num),
        )
        sample_indices = torch.empty_like(sample_labels)
        for i_klass, klass_index in enumerate(self.klass_indices):
            if klass_index.numel() == 0:
                continue
            left_inds = (sample_labels == i_klass).nonzero().squeeze(1)
            if len(left_inds) == 0:
                continue
            if self.element_weights is not None:
                right_inds = torch.multinomial(
                    self.element_weights[klass_index],
                    num_samples=len(klass_index)
                    if not self.replacement and len(klass_index) < len(left_inds)
                    else len(left_inds),
                    replacement=self.replacement,
                    generator=None
                    if self.restart_num == 0 and not self.modify_seed_on_requeue
                    else torch.Generator().manual_seed(self.restart_num),
                )
            elif self.replacement:
                right_inds = torch.randint(
                    len(klass_index),
                    size=(len(left_inds),),
                    generator=None
                    if self.restart_num == 0 and not self.modify_seed_on_requeue
                    else torch.Generator().manual_seed(self.restart_num),
                )
            else:
                maxelem = (
                    len(left_inds)
                    if len(left_inds) < len(klass_index)
                    else len(klass_index)
                )
                right_inds = torch.randperm(len(klass_index))[:maxelem]
            sample_indices[left_inds[: len(right_inds)]] = klass_index[right_inds]
            if len(right_inds) < len(left_inds):
                sample_indices[left_inds[len(right_inds) :]] = -1
        # drop all -1
        sample_indices = sample_indices[sample_indices != -1]
        # torch shuffle
        sample_indices = sample_indices[torch.randperm(len(sample_indices))]
        self.num_samples = len(sample_indices)
        # raise Exception("stop")
        yield from iter(sample_indices.tolist())

    def __len__(self):
        return self.num_samples
