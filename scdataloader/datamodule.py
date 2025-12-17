import math
import multiprocessing as mp
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from typing import List, Optional, Sequence, Union

import lamindb as ln
import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler, Subset
from torch.utils.data.sampler import (
    RandomSampler,
    SequentialSampler,
    SubsetRandomSampler,
    WeightedRandomSampler,
)
from tqdm import tqdm

from .collator import Collator
from .data import Dataset
from .utils import fileToList, getBiomartTable, listToFile

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
NNZ_SCALE = 1000


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        collection_name: str,
        clss_to_weight: List[str] = ["organism_ontology_term_id"],
        weight_scaler: int = 10,
        n_samples_per_epoch: int = 2_000_000,
        validation_split: float = 0.2,
        test_split: float = 0,
        use_default_col: bool = True,
        # this is for the mappedCollection
        clss_to_predict: List[str] = ["organism_ontology_term_id"],
        hierarchical_clss: List[str] = [],
        # this is for the collator
        how: str = "random expr",
        organism_col: str = "organism_ontology_term_id",
        max_len: int = 1000,
        replacement: bool = True,
        gene_subset: Optional[list[str]] = None,
        tp_name: Optional[str] = None,  # "heat_diff"
        assays_to_drop: List[str] = [
            # "EFO:0008853", #patch seq
            # "EFO:0010961", # visium
            "EFO:0030007",  # ATACseq
            # "EFO:0030062", # slide-seq
        ],
        metacell_mode: float = 0.0,
        get_knn_cells: bool = False,
        store_location: str = None,
        force_recompute_indices: bool = False,
        sampler_workers: int = None,
        sampler_chunk_size: int = None,
        organisms: Optional[str] = None,
        genedf: Optional[pd.DataFrame] = None,
        n_bins: int = 0,
        curiculum: int = 0,
        start_at: int = 0,
        **kwargs,
    ):
        """
        PyTorch Lightning DataModule for loading single-cell data from a LaminDB Collection.

        This DataModule provides train/val/test dataloaders with configurable sampling strategies.
        It combines MappedCollection, Dataset, and Collator to create efficient data pipelines
        for training single-cell foundation models.

        The training dataloader uses weighted random sampling based on class frequencies,
        validation uses random sampling, and test uses sequential sampling on held-out datasets.

        Args:
            collection_name (str): Key of the LaminDB Collection to load.
            clss_to_weight (List[str], optional): Label columns to use for weighted sampling
                in the training dataloader. Supports "nnz" for weighting by number of
                non-zero genes. Defaults to ["organism_ontology_term_id"].
            weight_scaler (int, optional): Controls balance between rare and common classes.
                Higher values lead to more uniform sampling across classes. Set to 0 to
                disable weighted sampling. Defaults to 10.
            n_samples_per_epoch (int, optional): Number of samples to draw per training epoch.
                Defaults to 2,000,000.
            validation_split (float | int, optional): Proportion (float) or absolute number (int)
                of samples for validation. Defaults to 0.2.
            test_split (float | int, optional): Proportion (float) or absolute number (int)
                of samples for testing. Uses entire datasets as test sets, rounding to
                nearest dataset boundary. Defaults to 0.
            use_default_col (bool, optional): Whether to use the default Collator for batch
                preparation. If False, no collate_fn is applied. Defaults to True.
            clss_to_predict (List[str], optional): Observation columns to encode as prediction
                targets. Must include "organism_ontology_term_id". Defaults to
                ["organism_ontology_term_id"].
            hierarchical_clss (List[str], optional): Observation columns with hierarchical
                ontology structure to be processed. Defaults to [].
            how (str, optional): Gene selection strategy passed to Collator. One of
                "most expr", "random expr", "all", "some". Defaults to "random expr".
            organism_col (str, optional): Column name for organism ontology term ID.
                Defaults to "organism_ontology_term_id".
            max_len (int, optional): Maximum number of genes per sample passed to Collator.
                Defaults to 1000.
            replacement (bool, optional): Whether to sample with replacement in training.
                Defaults to True.
            gene_subset (List[str], optional): List of genes to restrict the dataset to.
                Useful when model only supports specific genes. Defaults to None.
            tp_name (str, optional): Column name for time point or heat diffusion values.
                Defaults to None.
            assays_to_drop (List[str], optional): List of assay ontology term IDs to exclude
                from training. Defaults to ["EFO:0030007"] (ATAC-seq).
            metacell_mode (float, optional): Probability of using metacell aggregation mode.
                Cannot be used with get_knn_cells. Defaults to 0.0.
            get_knn_cells (bool, optional): Whether to include k-nearest neighbor cell
                expression data. Cannot be used with metacell_mode. Defaults to False.
            store_location (str, optional): Directory path to cache sampler indices and
                labels for faster subsequent loading. Defaults to None.
            force_recompute_indices (bool, optional): Force recomputation of cached indices
                even if they exist. Defaults to False.
            sampler_workers (int, optional): Number of parallel workers for building sampler
                indices. Auto-determined based on available CPUs if None. Defaults to None.
            sampler_chunk_size (int, optional): Chunk size for parallel sampler processing.
                Auto-determined based on available memory if None. Defaults to None.
            organisms (List[str], optional): List of organisms to include. If None, uses
                all organisms in the dataset. Defaults to None.
            genedf (pd.DataFrame, optional): Gene information DataFrame. If None, loaded
                automatically. Defaults to None.
            n_bins (int, optional): Number of bins for expression discretization. 0 means
                no binning. Defaults to 0.
            curiculum (int, optional): Curriculum learning parameter. If > 0, gradually
                increases sampling weight balance over epochs. Defaults to 0.
            start_at (int, optional): Starting index for resuming inference. Requires same
                number of GPUs as previous run. Defaults to 0.
            **kwargs: Additional arguments passed to PyTorch DataLoader (e.g., batch_size,
                num_workers, pin_memory).

        Attributes:
            dataset (Dataset): The underlying Dataset instance.
            classes (dict[str, int]): Mapping from class names to number of categories.
            train_labels (np.ndarray): Label array for weighted sampling.
            idx_full (np.ndarray): Indices for training samples.
            valid_idx (np.ndarray): Indices for validation samples.
            test_idx (np.ndarray): Indices for test samples.
            test_datasets (List[str]): Paths to datasets used for testing.

        Raises:
            ValueError: If "organism_ontology_term_id" not in clss_to_predict.
            ValueError: If both metacell_mode > 0 and get_knn_cells are True.

        Example:
            >>> dm = DataModule(
            ...     collection_name="my_collection",
            ...     batch_size=32,
            ...     num_workers=4,
            ...     max_len=2000,
            ... )
            >>> dm.setup()
            >>> train_loader = dm.train_dataloader()
        """
        if "organism_ontology_term_id" not in clss_to_predict:
            raise ValueError(
                "need 'organism_ontology_term_id' in the set of classes at least"
            )
        if metacell_mode > 0 and get_knn_cells:
            raise ValueError(
                "cannot use metacell mode and get_knn_cells at the same time"
            )
        mdataset = Dataset(
            ln.Collection.filter(key=collection_name, is_latest=True).first(),
            clss_to_predict=clss_to_predict,
            hierarchical_clss=hierarchical_clss,
            metacell_mode=metacell_mode,
            get_knn_cells=get_knn_cells,
            store_location=store_location,
            force_recompute_indices=force_recompute_indices,
            genedf=genedf,
        )
        # and location
        self.metacell_mode = bool(metacell_mode)
        self.gene_pos = None
        self.collection_name = collection_name
        if gene_subset is not None:
            tokeep = set(mdataset.genedf.index.tolist())
            gene_subset = [u for u in gene_subset if u in tokeep]
        self.classes = {k: len(v) for k, v in mdataset.class_topred.items()}
        # we might want not to order the genes by expression (or do it?)
        # we might want to not introduce zeros and

        if use_default_col:
            kwargs["collate_fn"] = Collator(
                organisms=mdataset.organisms if organisms is None else organisms,
                how=how,
                valid_genes=gene_subset,
                max_len=max_len,
                org_to_id=mdataset.encoder[organism_col],
                tp_name=tp_name,
                organism_name=organism_col,
                class_names=list(self.classes.keys()),
                genedf=genedf,
                n_bins=n_bins,
            )
        self.n_bins = n_bins
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
        self.n_samples_per_epoch = n_samples_per_epoch
        self.clss_to_weight = clss_to_weight
        self.train_weights = None
        self.train_labels = None
        self.sampler_workers = sampler_workers
        self.sampler_chunk_size = sampler_chunk_size
        self.store_location = store_location
        self.nnz = None
        self.start_at = start_at
        self.idx_full = None
        self.max_len = max_len
        self.test_datasets = []
        self.force_recompute_indices = force_recompute_indices
        self.curiculum = curiculum
        self.valid_idx = []
        self.test_idx = []
        super().__init__()
        print("finished init")

    def __repr__(self):
        return (
            f"DataLoader(\n"
            f"\twith a dataset=({self.dataset.__repr__()}\n)\n"
            f"\tvalidation_split={self.validation_split},\n"
            f"\ttest_split={self.test_split},\n"
            f"\tn_samples={self.n_samples},\n"
            f"\tweight_scaler={self.weight_scaler},\n"
            f"\tn_samples_per_epoch={self.n_samples_per_epoch},\n"
            f"\tassays_to_drop={self.assays_to_drop},\n"
            f"\tnum_datasets={len(self.dataset.mapped_dataset.storages)},\n"
            f"\ttest datasets={str(self.test_datasets)},\n"
            f"perc test: {str(len(self.test_idx) / self.n_samples)},\n"
            f"\tclss_to_weight={self.clss_to_weight}\n"
            + (
                "\twith train_dataset size of=(" + str(len(self.idx_full)) + ")\n)"
                if self.idx_full is not None
                else ")"
            )
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
    def genes_dict(self):
        return {
            i: self.dataset.genedf.index[self.dataset.genedf.organism == i].tolist()
            for i in self.dataset.organisms
        }

    def set_valid_genes_collator(self, genes):
        self.kwargs["collate_fn"]._setup(
            # cannot use genedf there since I am purposefully decreasing it...
            # genedf=self.dataset.genedf,
            org_to_id=self.kwargs["collate_fn"].org_to_id,
            valid_genes=genes,
        )

    @property
    def encoders(self):
        return self.dataset.encoder

    @encoders.setter
    def encoders(self, encoders):
        self.dataset.encoder = encoders
        self.kwargs["collate_fn"].org_to_id = encoders[
            self.kwargs["collate_fn"].organism_name
        ]
        self.kwargs["collate_fn"]._setup(
            org_to_id=self.kwargs["collate_fn"].org_to_id,
            valid_genes=self.genes,
        )

    @property
    def organisms(self):
        return self.dataset.organisms

    @organisms.setter
    def organisms(self, organisms):
        self.dataset.organisms = organisms
        self.kwargs["collate_fn"].organisms = organisms
        self.kwargs["collate_fn"]._setup(
            org_to_id=self.kwargs["collate_fn"].org_to_id,
            valid_genes=self.genes,
        )

    @property
    def num_datasets(self):
        return len(self.dataset.mapped_dataset.storages)

    def setup(self, stage=None):
        """
        Prepare data splits for training, validation, and testing.

        This method shuffles the data, computes sample weights for weighted sampling,
        removes samples from dropped assays, and creates train/val/test splits.
        Test splits use entire datasets to ensure evaluation on unseen data sources.

        Results can be cached to `store_location` for faster subsequent runs.

        Args:
            stage (str, optional): Training stage ('fit', 'test', or None for both).
                Currently not used but kept for Lightning compatibility. Defaults to None.

        Returns:
            List[str]: List of paths to test datasets.

        Note:
            Must be called before using dataloaders. The train/val/test split is
            deterministic when loading from cache.
        """
        print("setting up the datamodule")
        start_time = time.time()
        if (
            self.store_location is None
            or not os.path.exists(os.path.join(self.store_location, "train_labels.npy"))
            or self.force_recompute_indices
        ):
            if "nnz" in self.clss_to_weight and self.weight_scaler > 0:
                self.nnz = self.dataset.mapped_dataset.get_merged_labels(
                    "nnz", is_cat=False
                )
                self.clss_to_weight.remove("nnz")
                # Sigmoid scaling with 2 parameters
                midpoint = 2000
                steepness = 0.003
                # Apply sigmoid transformation
                # sigmoid(x) = 1 / (1 + exp(-steepness * (x - midpoint)))
                # Then scale to [1, NNZ_SCALE] range
                sigmoid_values = 1 / (1 + np.exp(-steepness * (self.nnz - midpoint)))
                self.nnz = 1 + ((NNZ_SCALE - 1) * sigmoid_values)
            if len(self.clss_to_weight) > 0 and self.weight_scaler > 0:
                labels = self.dataset.get_label_cats(
                    self.clss_to_weight,
                )
            else:
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
                    self.dataset.mapped_dataset.get_merged_labels(
                        "assay_ontology_term_id"
                    ),
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
                d_size = list(enumerate(self.dataset.mapped_dataset.n_obs_list))
                random.Random(42).shuffle(d_size)  # always same order
                for i, c in d_size:
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
            labels[~np.isin(np.arange(self.n_samples), idx_full)] = labels.max() + 1
            # some labels will now not exist anymore as replaced by len(weights) - 1.
            # this means that the associated weights should be 0.
            # by doing np.bincount(labels)*weights this will be taken into account
            self.train_labels = labels
            self.idx_full = idx_full
        if self.store_location is not None:
            if (
                not os.path.exists(
                    os.path.join(self.store_location, "train_labels.npy")
                )
                or self.force_recompute_indices
            ):
                os.makedirs(self.store_location, exist_ok=True)
                if self.nnz is not None:
                    np.save(os.path.join(self.store_location, "nnz.npy"), self.nnz)
                np.save(
                    os.path.join(self.store_location, "train_labels.npy"),
                    self.train_labels,
                )
                np.save(
                    os.path.join(self.store_location, "idx_full.npy"), self.idx_full
                )
                if self.test_idx is not None:
                    np.save(
                        os.path.join(self.store_location, "test_idx.npy"), self.test_idx
                    )
                if self.valid_idx is not None:
                    np.save(
                        os.path.join(self.store_location, "valid_idx.npy"),
                        self.valid_idx,
                    )
                listToFile(
                    self.test_datasets,
                    os.path.join(self.store_location, "test_datasets.txt"),
                )
            else:
                self.nnz = (
                    np.load(os.path.join(self.store_location, "nnz.npy"), mmap_mode="r")
                    if os.path.exists(os.path.join(self.store_location, "nnz.npy"))
                    else None
                )
                self.train_labels = np.load(
                    os.path.join(self.store_location, "train_labels.npy")
                )
                self.idx_full = np.load(
                    os.path.join(self.store_location, "idx_full.npy"), mmap_mode="r"
                )
                self.test_idx = (
                    np.load(os.path.join(self.store_location, "test_idx.npy"))
                    if os.path.exists(os.path.join(self.store_location, "test_idx.npy"))
                    else None
                )
                self.valid_idx = (
                    np.load(os.path.join(self.store_location, "valid_idx.npy"))
                    if os.path.exists(
                        os.path.join(self.store_location, "valid_idx.npy")
                    )
                    else None
                )
                self.test_datasets = fileToList(
                    os.path.join(self.store_location, "test_datasets.txt")
                )
                print("loaded from store")
        print(f"done setup, took {time.time() - start_time:.2f} seconds")
        return self.test_datasets

    def train_dataloader(self, **kwargs):
        """
        Create the training DataLoader with weighted random sampling.

        Uses LabelWeightedSampler for class-balanced sampling when weight_scaler > 0
        and clss_to_weight is specified. Otherwise uses RankShardSampler for
        distributed training without weighting.

        Args:
            **kwargs: Additional arguments passed to DataLoader, overriding defaults.

        Returns:
            DataLoader: Training DataLoader instance.

        Raises:
            ValueError: If setup() has not been called.
        """
        if len(self.clss_to_weight) > 0 and self.weight_scaler > 0:
            try:
                print("Setting up the parallel train sampler...")
                # Create the optimized parallel sampler
                print(f"Using {self.sampler_workers} workers for class indexing")
                train_sampler = LabelWeightedSampler(
                    labels=self.train_labels,
                    weight_scaler=self.weight_scaler,
                    num_samples=int(self.n_samples_per_epoch),
                    element_weights=self.nnz,
                    replacement=self.replacement,
                    n_workers=self.sampler_workers,
                    chunk_size=self.sampler_chunk_size,
                    store_location=self.store_location,
                    force_recompute_indices=self.force_recompute_indices,
                    curiculum=self.curiculum,
                )
            except ValueError as e:
                raise ValueError(str(e) + " Have you run `datamodule.setup()`?")
            dataset = None
        else:
            dataset = Subset(self.dataset, self.idx_full)
            train_sampler = RankShardSampler(len(dataset), start_at=self.start_at)
        current_loader_kwargs = kwargs.copy()
        current_loader_kwargs.update(self.kwargs)
        return DataLoader(
            self.dataset if dataset is None else dataset,
            sampler=train_sampler,
            **current_loader_kwargs,
        )

    def val_dataloader(self):
        """
        Create the validation DataLoader.

        Returns:
            DataLoader | List: Validation DataLoader, or empty list if no validation split.
        """
        return (
            DataLoader(
                Subset(self.dataset, self.valid_idx),
                **self.kwargs,
            )
            if self.valid_idx is not None
            else []
        )

    def test_dataloader(self):
        """
        Create the test DataLoader with sequential sampling.

        Returns:
            DataLoader | List: Test DataLoader, or empty list if no test split.
        """
        return (
            DataLoader(
                self.dataset, sampler=SequentialSampler(self.test_idx), **self.kwargs
            )
            if self.test_idx is not None
            else []
        )

    def predict_dataloader(self):
        """
        Create a DataLoader for prediction over all training data.

        Uses RankShardSampler for distributed inference.

        Returns:
            DataLoader: Prediction DataLoader instance.
        """
        subset = Subset(self.dataset, self.idx_full)
        return DataLoader(
            self.dataset,
            sampler=RankShardSampler(len(subset), start_at=self.start_at),
            **self.kwargs,
        )


class LabelWeightedSampler(Sampler[int]):

    label_weights: torch.Tensor
    klass_indices: dict[int, torch.Tensor]
    num_samples: int
    element_weights: Optional[torch.Tensor]
    replacement: bool

    def __init__(
        self,
        labels: np.ndarray,
        num_samples: int,
        replacement: bool = True,
        weight_scaler: Optional[float] = None,
        element_weights: Optional[Sequence[float]] = None,
        n_workers: int = None,
        chunk_size: int = None,  # Process 10M elements per chunk
        store_location: str = None,
        force_recompute_indices: bool = False,
        curiculum: int = 0,
    ) -> None:
        """
        Weighted random sampler balancing both class frequencies and element weights.

        This sampler is optimized for very large datasets (millions of samples) with:
        - Parallel construction of class indices using multiple CPU workers
        - Chunked processing to manage memory usage
        - Support for curriculum learning via progressive weight scaling
        - Optional per-element weights (e.g., by number of expressed genes)

        The sampling process:
        1. Sample class labels according to class weights
        2. For each sampled class, sample elements according to element weights
        3. Shuffle all sampled indices

        Args:
            labels (np.ndarray): Integer class label for each dataset element.
                Shape: (dataset_size,). The last unique label is treated as
                "excluded" with weight 0.
            num_samples (int): Number of samples to draw per epoch.
            replacement (bool, optional): Whether to sample with replacement.
                Defaults to True.
            weight_scaler (float, optional): Controls class weight balance.
                Weight formula: (scaler * count) / (count + scaler).
                Higher values = more uniform sampling. Defaults to None.
            element_weights (Sequence[float], optional): Per-element sampling weights.
                Shape: (dataset_size,). Defaults to None (uniform within class).
            n_workers (int, optional): Number of parallel workers for index building.
                Defaults to min(20, num_cpus - 1).
            chunk_size (int, optional): Elements per chunk for parallel processing.
                Auto-determined based on available memory if None.
            store_location (str, optional): Directory to cache computed indices.
                Defaults to None.
            force_recompute_indices (bool, optional): Recompute indices even if cached.
                Defaults to False.
            curiculum (int, optional): Curriculum learning epochs. If > 0, weight
                exponent increases from 0 to 1 over this many epochs. Defaults to 0.

        Attributes:
            label_weights (torch.Tensor): Computed weights per class label.
            klass_indices (torch.Tensor): Concatenated indices for all classes.
            klass_offsets (torch.Tensor): Starting offset for each class in klass_indices.
            count (int): Number of times __iter__ has been called (for curriculum).

        Example:
            >>> sampler = LabelWeightedSampler(
            ...     labels=train_labels,
            ...     num_samples=1_000_000,
            ...     weight_scaler=10,
            ...     element_weights=nnz_weights,
            ... )
            >>> for idx in sampler:
            ...     # Process sample at idx
            ...     pass
        """
        print("Initializing optimized parallel weighted sampler...")
        super(LabelWeightedSampler, self).__init__(None)
        self.count = 0
        self.curiculum = curiculum

        # Compute label weights (incorporating class frequencies)
        # Directly use labels as numpy array without conversion
        counts = np.bincount(labels)
        counts[-1] = 0  # Ensure the weight for the 'NONE' class is zero
        label_weights = (weight_scaler * counts) / (counts + weight_scaler)
        self.label_weights = torch.as_tensor(
            label_weights, dtype=torch.float32
        ).share_memory_()

        # Store element weights if provided
        if element_weights is not None:
            self.element_weights = torch.as_tensor(
                element_weights, dtype=torch.float32
            ).share_memory_()
        else:
            self.element_weights = None

        self.replacement = replacement
        self.num_samples = num_samples
        if (
            store_location is None
            or not os.path.exists(os.path.join(store_location, "klass_indices.pt"))
            or force_recompute_indices
        ):
            # Set number of workers (default to CPU count - 1, but at least 1)
            if n_workers is None:
                # Check if running on SLURM
                n_workers = min(20, max(1, mp.cpu_count() - 1))
                if "SLURM_CPUS_PER_TASK" in os.environ:
                    n_workers = min(
                        20, max(1, int(os.environ["SLURM_CPUS_PER_TASK"]) - 1)
                    )

            # Try to auto-determine optimal chunk size based on memory
            if chunk_size is None:
                try:
                    import psutil

                    # Check if running on SLURM
                    available_memory = psutil.virtual_memory().available
                    for name in [
                        "SLURM_MEM_PER_NODE",
                        "SLURM_MEM_PER_CPU",
                        "SLURM_MEM_PER_GPU",
                        "SLURM_MEM_PER_TASK",
                    ]:
                        if name in os.environ:
                            available_memory = (
                                int(os.environ[name]) * 1024 * 1024
                            )  # Convert MB to bytes
                            break

                    # Use at most 50% of available memory across all workers
                    memory_per_worker = 0.5 * available_memory / n_workers
                    # Rough estimate: each label takes 4 bytes, each index 8 bytes
                    bytes_per_element = 12
                    chunk_size = min(
                        max(100_000, int(memory_per_worker / bytes_per_element / 3)),
                        2_000_000,
                    )
                    print(f"Auto-determined chunk size: {chunk_size:,} elements")
                except (ImportError, KeyError):
                    chunk_size = 2_000_000
                    print(f"Using default chunk size: {chunk_size:,} elements")

            # Parallelize the class indices building
            print(f"Building class indices in parallel with {n_workers} workers...")
            klass_indices = self._build_class_indices_parallel(
                labels, chunk_size, n_workers
            )

            # Convert klass_indices to a single tensor and offset vector
            all_indices = []
            offsets = []
            current_offset = 0

            # Sort keys to ensure consistent ordering
            keys = klass_indices.keys()

            # Build concatenated tensor and track offsets
            for i in range(max(keys) + 1):
                offsets.append(current_offset)
                if i in keys:
                    indices = klass_indices[i]
                    all_indices.append(indices)
                    current_offset += len(indices)

            # Convert to tensors
            self.klass_indices = torch.cat(all_indices).to(torch.int32).share_memory_()
            self.klass_offsets = torch.tensor(offsets, dtype=torch.long).share_memory_()
        if store_location is not None:
            store_path = os.path.join(store_location, "klass_indices.pt")
            if os.path.exists(store_path) and not force_recompute_indices:
                self.klass_indices = torch.load(store_path).share_memory_()
                self.klass_offsets = torch.load(
                    store_path.replace(".pt", "_offsets.pt")
                ).share_memory_()
                print(f"Loaded sampler indices from {store_path}")
            else:
                torch.save(self.klass_indices, store_path)
                torch.save(self.klass_offsets, store_path.replace(".pt", "_offsets.pt"))
                print(f"Saved sampler indices to {store_path}")
        print(f"Done initializing sampler with {len(self.klass_offsets)} classes")

    def __iter__(self):
        self.count += 1
        # Sample classes according to their weights
        print("sampling a new batch of size", self.num_samples)

        sample_labels = torch.multinomial(
            (
                self.label_weights ** min(1, ((self.count + 5) / self.curiculum))
                if self.curiculum
                else self.label_weights
            ),
            num_samples=self.num_samples,
            replacement=True,
        )
        # Get counts of each class in sample_labels
        unique_samples, sample_counts = torch.unique(sample_labels, return_counts=True)

        # Initialize result tensor
        result_indices_list = (
            []
        )  # Changed name to avoid conflict if you had result_indices elsewhere

        # Process only the classes that were actually sampled
        for i, (label, count) in tqdm(
            enumerate(zip(unique_samples.tolist(), sample_counts.tolist())),
            total=len(unique_samples),
            desc="Processing classes in sampler",
        ):
            klass_index = self.klass_indices[
                self.klass_offsets[label] : self.klass_offsets[label + 1]
            ]

            if klass_index.numel() == 0:
                continue

            # Sample elements from this class
            if self.element_weights is not None:
                # This is a critical point for memory
                current_element_weights_slice = self.element_weights[klass_index]

                if current_element_weights_slice.shape[0] >= (2**24) - 1:
                    ind = torch.randperm(len(klass_index))[: (2**24) - 10]
                    klass_index = klass_index[ind]
                    current_element_weights_slice = current_element_weights_slice[ind]

                if self.replacement:
                    right_inds = torch.multinomial(
                        current_element_weights_slice,
                        num_samples=count,
                        replacement=True,
                    )
                else:
                    num_to_sample = min(count, len(klass_index))
                    right_inds = torch.multinomial(
                        current_element_weights_slice,
                        num_samples=num_to_sample,
                        replacement=False,
                    )
            elif self.replacement:
                right_inds = torch.randint(len(klass_index), size=(count,))
            else:
                num_to_sample = min(count, len(klass_index))
                right_inds = torch.randperm(len(klass_index))[:num_to_sample]

            # Get actual indices
            sampled_indices = klass_index[right_inds]
            result_indices_list.append(sampled_indices)

        # Combine all indices
        if result_indices_list:  # Check if the list is not empty
            final_result_indices = torch.cat(
                result_indices_list
            )  # Use the list with the appended new name

            # Shuffle the combined indices
            shuffled_indices = final_result_indices[
                torch.randperm(len(final_result_indices))
            ]
            self.num_samples = len(shuffled_indices)
            yield from shuffled_indices.tolist()
        else:
            self.num_samples = 0
            yield from iter([])

    def __len__(self):
        return self.num_samples

    def _merge_chunk_results(self, results_list):
        """Merge results from multiple chunks into a single dictionary.

        Args:
            results_list: list of dictionaries mapping class labels to index arrays

        Returns:
            merged dictionary with PyTorch tensors
        """
        merged = {}

        # Collect all labels across all chunks
        all_labels = set()
        for chunk_result in results_list:
            all_labels.update(chunk_result.keys())

        # For each unique label
        for label in all_labels:
            # Collect indices from all chunks where this label appears
            indices_lists = [
                chunk_result[label]
                for chunk_result in results_list
                if label in chunk_result
            ]

            if indices_lists:
                # Concatenate all indices for this label
                merged[label] = torch.tensor(
                    np.concatenate(indices_lists), dtype=torch.long
                )
            else:
                merged[label] = torch.tensor([], dtype=torch.long)

        return merged

    def _build_class_indices_parallel(self, labels, chunk_size, n_workers=None):
        """Build class indices in parallel across multiple workers.

        Args:
            labels: array of class labels
            n_workers: number of parallel workers
            chunk_size: size of chunks to process

        Returns:
            dictionary mapping class labels to tensors of indices
        """
        n = len(labels)
        results = []
        # Create chunks of the labels array with proper sizing
        n_chunks = (n + chunk_size - 1) // chunk_size  # Ceiling division
        print(f"Processing {n:,} elements in {n_chunks} chunks...")

        # Process in chunks to limit memory usage
        with ProcessPoolExecutor(
            max_workers=n_workers, mp_context=mp.get_context("spawn")
        ) as executor:
            # Submit chunks for processing
            futures = []
            for i in range(n_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, n)
                # We pass only chunk boundaries, not the data itself
                # This avoids unnecessary copies during process creation
                futures.append(
                    executor.submit(
                        self._process_chunk_with_slice,
                        (start_idx, end_idx, labels),
                    )
                )

            # Collect results as they complete with progress reporting
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing chunks"
            ):
                results.append(future.result())

        # Merge results from all chunks
        print("Merging results from all chunks...")
        merged_results = self._merge_chunk_results(results)

        return merged_results

    def _process_chunk_with_slice(self, slice_info):
        """Process a slice of the labels array by indices.

        Args:
            slice_info: tuple of (start_idx, end_idx, labels_array) where
                       start_idx and end_idx define the slice to process

        Returns:
            dict mapping class labels to arrays of indices
        """
        start_idx, end_idx, labels_array = slice_info

        # We're processing a slice of the original array
        labels_slice = labels_array[start_idx:end_idx]
        chunk_indices = {}

        # Create a direct map of indices
        indices = np.arange(start_idx, end_idx)

        # Get unique labels in this slice for more efficient processing
        unique_labels = np.unique(labels_slice)
        # For each valid label, find its indices
        for label in unique_labels:
            # Find positions where this label appears (using direct boolean indexing)
            label_mask = labels_slice == label
            chunk_indices[int(label)] = indices[label_mask]

        return chunk_indices


class RankShardSampler(Sampler[int]):
    """
    Sampler that shards data contiguously across distributed ranks.

    Divides the dataset into contiguous chunks, one per rank, without
    padding or duplicating samples. Preserves the original data order
    within each shard (useful for pre-shuffled data).

    Args:
        data_len (int): Total number of samples in the dataset.
        start_at (int, optional): Global starting index for resuming training.
            Requires the same number of GPUs as the previous run. Defaults to 0.

    Attributes:
        rank (int): Current process rank (0 if not distributed).
        world_size (int): Total number of processes (1 if not distributed).
        start (int): Starting index for this rank's shard.
        end (int): Ending index (exclusive) for this rank's shard.

    Note:
        The last rank may have fewer samples than others if the dataset
        size is not evenly divisible by world_size.

    Example:
        >>> sampler = RankShardSampler(len(dataset))
        >>> loader = DataLoader(dataset, sampler=sampler)
    """

    def __init__(self, data_len: int, start_at: int = 0) -> None:
        self.data_len = data_len
        self.start_at = start_at
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
        else:
            self.rank, self.world_size = 0, 1

        # contiguous chunk per rank (last rank may be shorter)
        if self.start_at > 0:
            print(
                "!!!!ATTENTION: make sure that you are running on the exact same \
                number of GPU as your previous run!!!!!"
            )
        print(f"Sharding data of size {data_len} over {self.world_size} ranks")
        per_rank = math.ceil(self.data_len / self.world_size)
        self.start = int((self.start_at / self.world_size) + (self.rank * per_rank))
        self.end = min((self.rank + 1) * per_rank, self.data_len)
        print(f"Rank {self.rank} processing indices from {self.start} to {self.end}")

    def __iter__(self):
        return iter(range(self.start, self.end))

    def __len__(self):
        return self.end - self.start
