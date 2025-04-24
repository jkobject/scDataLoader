import os
from typing import Optional, Sequence, Union
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

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
from .utils import getBiomartTable
import random
from tqdm import tqdm

FILE_DIR = os.path.dirname(os.path.abspath(__file__))


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        collection_name: str,
        clss_to_weight: list = ["organism_ontology_term_id"],
        weight_scaler: int = 10,
        n_samples_per_epoch: int = 2_000_000,
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
            n_samples_per_epoch (int, optional): The number of samples to include in the training set for each epoch. Defaults to 2_000_000.
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
            get_knn_cells (bool, optional): Whether to get the k-nearest neighbors of each queried cells. Defaults to False.
            **kwargs: Additional keyword arguments passed to the pytorch DataLoader.
            see @file data.py and @file collator.py for more details about some of the parameters
        """
        if "organism_ontology_term_id" not in clss_to_predict:
            raise ValueError(
                "need 'organism_ontology_term_id' in the set of classes at least"
            )
        mdataset = Dataset(
            ln.Collection.filter(name=collection_name, is_latest=True).first(),
            clss_to_predict=clss_to_predict,
            hierarchical_clss=hierarchical_clss,
            metacell_mode=metacell_mode,
            get_knn_cells=get_knn_cells,
        )
        # and location
        organisms = mdataset.class_topred["organism_ontology_term_id"]
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
                pd.read_parquet(gene_embeddings).loc[:, :2], how="inner"
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
        self.organisms = organisms
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
        self.nnz = None
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
            f"\tn_samples_per_epoch={self.n_samples_per_epoch},\n"
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
        print("setting up the datamodule")
        if "nnz" in self.clss_to_weight and self.weight_scaler > 0:
            self.nnz = self.dataset.mapped_dataset.get_merged_labels(
                "nnz", is_cat=False
            )
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
        weights = np.concatenate([weights, np.zeros(1)])
        labels[~np.isin(np.arange(self.n_samples), idx_full)] = len(weights) - 1
        # some labels will now not exist anymore as replaced by len(weights) - 1.
        # this means that the associated weights should be 0.
        # by doing np.bincount(labels)*weights this will be taken into account
        self.train_weights = weights
        self.train_labels = labels
        self.idx_full = idx_full
        print("done setup")
        return self.test_datasets

    def train_dataloader(self, **kwargs):
        try:
            print("Setting up the parallel train sampler...")

            # Get number of workers from kwargs, environment variable, or use a reasonable default
            n_workers = kwargs.pop(
                "sampler_workers",
                int(os.environ.get("SAMPLER_WORKERS", max(1, mp.cpu_count() - 1))),
            )

            # Let the sampler auto-determine chunk size
            chunk_size = kwargs.pop("sampler_chunk_size", None)

            # Create the optimized parallel sampler
            print(f"Using {n_workers} workers for class indexing")
            train_sampler = LabelWeightedSampler(
                label_weights=self.train_weights,
                labels=self.train_labels,
                num_samples=int(self.n_samples_per_epoch),
                element_weights=self.nnz,
                replacement=self.replacement,
                n_workers=n_workers,
                chunk_size=chunk_size,
            )
        except ValueError as e:
            raise ValueError(str(e) + " Have you run `datamodule.setup()`?")

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

    @staticmethod
    def _estimate_optimal_chunk_size(labels_size, n_workers, max_memory_fraction=0.5):
        """Estimate optimal chunk size based on system memory.

        Args:
            labels_size: Size of the labels array
            n_workers: Number of worker processes
            max_memory_fraction: Maximum fraction of system memory to use

        Returns:
            Recommended chunk size
        """
        try:
            import psutil

            # Get available system memory in bytes
            available_memory = psutil.virtual_memory().available

            # Calculate bytes per element (int32 = 4 bytes)
            bytes_per_element = 4

            # Calculate memory per worker
            memory_per_worker = available_memory * max_memory_fraction / n_workers

            # Limit chunk size based on memory per worker
            # We need memory for:
            # 1. The chunk itself
            # 2. The indices created
            # 3. Other overhead (sorting, etc.) - factor of 3
            memory_per_chunk = memory_per_worker / 3
            max_elements_per_chunk = int(memory_per_chunk / (bytes_per_element + 8))

            # Ensure a minimum chunk size
            min_chunk_size = 1_000_000
            chunk_size = max(min_chunk_size, max_elements_per_chunk)

            # Ensure chunk size is reasonable for the dataset
            chunk_size = min(chunk_size, labels_size // n_workers + 1)

            print(f"Automatically determined chunk size: {chunk_size:,} elements")
            return chunk_size

        except ImportError:
            # If psutil is not available, use a reasonable default
            default_size = 10_000_000
            print(f"psutil not available, using default chunk size: {default_size:,}")
            return default_size


class LabelWeightedSampler(Sampler[int]):
    """
    A weighted random sampler that samples from a dataset with respect to both class weights and element weights.

    This sampler is designed to handle very large datasets efficiently, with optimizations for:
    1. Parallel building of class indices
    2. Chunked processing for large arrays
    3. Efficient memory management
    4. Proper handling of replacement and non-replacement sampling
    """

    label_weights: torch.Tensor
    klass_indices: dict[int, torch.Tensor]
    num_samples: int
    element_weights: Optional[torch.Tensor]
    replacement: bool

    def __init__(
        self,
        label_weights: Sequence[float],
        labels: np.ndarray,
        num_samples: int,
        replacement: bool = True,
        element_weights: Optional[Sequence[float]] = None,
        n_workers: int = None,
        chunk_size: int = 10_000_000,  # Process 10M elements per chunk
    ) -> None:
        """
        Initialize the sampler with parallel processing for large datasets.

        Args:
            label_weights: Weights for each class (length = number of classes)
            labels: Class label for each dataset element (length = dataset size)
            num_samples: Number of samples to draw
            replacement: Whether to sample with replacement
            element_weights: Optional weights for each element within classes
            n_workers: Number of parallel workers to use (default: number of CPUs-1)
            chunk_size: Size of chunks to process in parallel (default: 10M elements)
        """
        print("Initializing optimized parallel weighted sampler...")
        super(LabelWeightedSampler, self).__init__(None)

        # Compute label weights (incorporating class frequencies)
        # Directly use labels as numpy array without conversion
        label_weights = np.asarray(label_weights) * np.bincount(labels)
        self.label_weights = torch.as_tensor(label_weights, dtype=torch.float32)

        # Store element weights if provided
        if element_weights is not None:
            self.element_weights = torch.as_tensor(element_weights, dtype=torch.float32)
        else:
            self.element_weights = None

        self.replacement = replacement
        self.num_samples = num_samples

        # Set number of workers (default to CPU count - 1, but at least 1)
        if n_workers is None:
            # Check if running on SLURM
            n_workers = min(20, max(1, mp.cpu_count() - 1))
            if "SLURM_CPUS_PER_TASK" in os.environ:
                n_workers = min(20, max(1, int(os.environ["SLURM_CPUS_PER_TASK"]) - 1))

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
                    max(1_000_000, int(memory_per_worker / bytes_per_element / 3)),
                    5_000_000,
                )
                print(f"Auto-determined chunk size: {chunk_size:,} elements")
            except (ImportError, KeyError):
                chunk_size = 5_000_000
                print(f"Using default chunk size: {chunk_size:,} elements")

        # Parallelize the class indices building
        print(f"Building class indices in parallel with {n_workers} workers...")
        self.klass_indices = self._build_class_indices_parallel(
            labels, chunk_size, n_workers
        )

        # Calculate class sizes
        self.klass_sizes = {k: len(v) for k, v in self.klass_indices.items()}

        print(f"Done initializing sampler with {len(self.klass_indices)} classes")

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
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
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

    def __iter__(self):
        # Sample classes according to their weights
        sample_labels = torch.multinomial(
            self.label_weights,
            num_samples=self.num_samples,
            replacement=True,
        )
        print("sampling a new batch of size", self.num_samples)
        # Get counts of each class in sample_labels
        unique_samples, sample_counts = torch.unique(sample_labels, return_counts=True)

        # Initialize result tensor
        result_indices = []

        # Process only the classes that were actually sampled
        for label, count in zip(unique_samples.tolist(), sample_counts.tolist()):
            if label not in self.klass_indices:
                continue

            klass_index = self.klass_indices[label]
            if klass_index.numel() == 0:
                continue

            # Sample elements from this class
            if self.element_weights is not None:
                # Use element weights for this class
                if self.replacement:
                    # With replacement - can sample as many as needed
                    right_inds = torch.multinomial(
                        self.element_weights[klass_index],
                        num_samples=count,
                        replacement=True,
                    )
                else:
                    # Without replacement - can only sample up to class size
                    num_to_sample = min(count, len(klass_index))
                    right_inds = torch.multinomial(
                        self.element_weights[klass_index],
                        num_samples=num_to_sample,
                        replacement=False,
                    )
            elif self.replacement:
                # Random sampling with replacement
                right_inds = torch.randint(len(klass_index), size=(count,))
            else:
                # Without replacement - can only sample up to class size
                num_to_sample = min(count, len(klass_index))
                right_inds = torch.randperm(len(klass_index))[:num_to_sample]

            # Get actual indices
            sampled_indices = klass_index[right_inds]
            result_indices.append(sampled_indices)

        # Combine all indices
        if result_indices:
            combined_indices = torch.cat(result_indices)

            # Shuffle the combined indices
            shuffled_indices = combined_indices[torch.randperm(len(combined_indices))]
            self.num_samples = len(shuffled_indices)
            yield from shuffled_indices.tolist()
        else:
            self.num_samples = 0
            yield from iter([])

    def __len__(self):
        return self.num_samples
