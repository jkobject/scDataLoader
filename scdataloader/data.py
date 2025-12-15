import warnings
from collections import Counter
from dataclasses import dataclass, field
from functools import reduce
from typing import List, Literal, Optional, Union

# ln.connect("scprint")
import bionty as bt
import lamindb as ln
import numpy as np
import pandas as pd
from anndata import AnnData
from lamindb.core.storage._anndata_accessor import _safer_read_index
from scipy.sparse import issparse
from torch.utils.data import Dataset as torchDataset

from scdataloader.utils import get_ancestry_mapping, load_genes

from .mapped import MappedCollection, _Connect


@dataclass
class Dataset(torchDataset):
    """
    PyTorch Dataset for loading single-cell data from a LaminDB Collection.

    This class wraps LaminDB's MappedCollection to provide additional features:
    - Management of hierarchical ontology labels (cell type, tissue, disease, etc.)
    - Automatic encoding of categorical labels to integers
    - Multi-species gene handling with unified gene indexing
    - Optional metacell aggregation and KNN neighbor retrieval

    The dataset lazily loads data from storage, making it memory-efficient for
    large collections spanning multiple files.

    Args:
        lamin_dataset (ln.Collection): LaminDB Collection containing the artifacts to load.
        genedf (pd.DataFrame, optional): DataFrame with gene information, indexed by gene ID
            with an 'organism' column. If None, automatically loaded based on organisms
            in the dataset. Defaults to None.
        clss_to_predict (List[str], optional): Observation columns to encode as prediction
            targets. These will be integer-encoded in the output. Defaults to [].
        hierarchical_clss (List[str], optional): Observation columns with hierarchical
            ontology structure. These will have their ancestry relationships computed
            using Bionty. Supported columns:
            - "cell_type_ontology_term_id"
            - "tissue_ontology_term_id"
            - "disease_ontology_term_id"
            - "development_stage_ontology_term_id"
            - "assay_ontology_term_id"
            - "self_reported_ethnicity_ontology_term_id"
            Defaults to [].
        join_vars (str, optional): How to join variables across artifacts.
            "inner" for intersection, "outer" for union, None for no joining.
            Defaults to None.
        metacell_mode (float, optional): Probability of returning aggregated metacell
            expression instead of single-cell. Defaults to 0.0.
        get_knn_cells (bool, optional): Whether to include k-nearest neighbor cell
            expression in the output. Requires precomputed neighbors in the data.
            Defaults to False.
        store_location (str, optional): Directory path to cache computed indices.
            Defaults to None.
        force_recompute_indices (bool, optional): Force recomputation of cached data.
            Defaults to False.

    Attributes:
        mapped_dataset (MappedCollection): Underlying mapped collection for data access.
        genedf (pd.DataFrame): Gene information DataFrame.
        organisms (List[str]): List of organism ontology term IDs in the dataset.
        class_topred (dict[str, set]): Mapping from class name to set of valid labels.
        labels_groupings (dict[str, dict]): Hierarchical groupings for ontology classes.
        encoder (dict[str, dict]): Label encoders mapping strings to integers.

    Raises:
        ValueError: If genedf is None and "organism_ontology_term_id" is not in clss_to_predict.

    Example:
        >>> collection = ln.Collection.filter(key="my_collection").first()
        >>> dataset = Dataset(
        ...     lamin_dataset=collection,
        ...     clss_to_predict=["organism_ontology_term_id", "cell_type_ontology_term_id"],
        ...     hierarchical_clss=["cell_type_ontology_term_id"],
        ... )
        >>> sample = dataset[0]  # Returns dict with "X" and encoded labels
    """

    lamin_dataset: ln.Collection
    genedf: Optional[pd.DataFrame] = None
    # set of obs to prepare for prediction (encode)
    clss_to_predict: Optional[List[str]] = field(default_factory=list)
    # set of obs that need to be hierarchically prepared
    hierarchical_clss: Optional[List[str]] = field(default_factory=list)
    join_vars: Literal["inner", "outer"] | None = None
    metacell_mode: float = 0.0
    get_knn_cells: bool = False
    store_location: str | None = None
    force_recompute_indices: bool = False

    def __post_init__(self):
        # see at the end of the file for the mapped function
        self.mapped_dataset = mapped(
            self.lamin_dataset,
            obs_keys=list(set(self.hierarchical_clss + self.clss_to_predict)),
            join=self.join_vars,
            encode_labels=self.clss_to_predict,
            unknown_label="unknown",
            stream=True,
            parallel=True,
            metacell_mode=self.metacell_mode,
            get_knn_cells=self.get_knn_cells,
            store_location=self.store_location,
            force_recompute_indices=self.force_recompute_indices,
        )
        print(
            "won't do any check but we recommend to have your dataset coming from local storage"
        )
        self.labels_groupings = {}
        self.class_topred = {}
        # generate tree from ontologies
        if len(self.hierarchical_clss) > 0:
            self.define_hierarchies(self.hierarchical_clss)
        if len(self.clss_to_predict) > 0:
            for clss in self.clss_to_predict:
                if clss not in self.hierarchical_clss:
                    # otherwise it's already been done
                    self.class_topred[clss] = set(
                        self.mapped_dataset.encoders[clss].keys()
                    )
                    if (
                        self.mapped_dataset.unknown_label
                        in self.mapped_dataset.encoders[clss].keys()
                    ):
                        self.class_topred[clss] -= set(
                            [self.mapped_dataset.unknown_label]
                        )
        if self.genedf is None:
            if "organism_ontology_term_id" not in self.clss_to_predict:
                raise ValueError(
                    "need 'organism_ontology_term_id' in the set of classes if you don't provide a genedf"
                )
            self.organisms = list(self.class_topred["organism_ontology_term_id"])
            self.genedf = load_genes(self.organisms)
        else:
            self.organisms = self.genedf["organism"].unique().tolist()
        self.organisms.sort()

        self.genedf.columns = self.genedf.columns.astype(str)
        # self.check_aligned_vars()

    def check_aligned_vars(self):
        vars = self.genedf.index.tolist()
        i = 0
        for storage in self.mapped_dataset.storages:
            with _Connect(storage) as store:
                if len(set(_safer_read_index(store["var"]).tolist()) - set(vars)) == 0:
                    i += 1
        print("{}% are aligned".format(i * 100 / len(self.mapped_dataset.storages)))

    def __len__(self, **kwargs):
        return self.mapped_dataset.__len__(**kwargs)

    @property
    def encoder(self):
        return self.mapped_dataset.encoders

    @encoder.setter
    def encoder(self, encoder):
        self.mapped_dataset.encoders = encoder

    def __getitem__(self, *args, **kwargs):
        item = self.mapped_dataset.__getitem__(*args, **kwargs)
        return item

    def __repr__(self):
        return (
            "total dataset size is {} Gb\n".format(
                sum([file.size for file in self.lamin_dataset.artifacts.all()]) / 1e9
            )
            + "---\n"
            + "dataset contains:\n"
            + "     {} cells\n".format(self.mapped_dataset.__len__())
            + "     {} genes\n".format(self.genedf.shape[0])
            + "     {} clss_to_predict\n".format(len(self.clss_to_predict))
            + "     {} hierarchical_clss\n".format(len(self.hierarchical_clss))
            + (
                "     {} organisms\n".format(len(self.organisms))
                if self.organisms is not None
                else ""
            )
            + (
                "dataset contains {} classes to predict\n".format(
                    sum([len(self.class_topred[i]) for i in self.class_topred])
                )
                if len(self.class_topred) > 0
                else ""
            )
            + "     {} metacell_mode\n".format(self.metacell_mode)
        )

    def get_label_cats(
        self,
        obs_keys: Union[str, List[str]],
    ):
        """
        Get combined categorical codes for one or more label columns.

        Retrieves labels from the mapped dataset and combines them into a single
        categorical encoding. Useful for creating compound class labels for
        stratified sampling.

        Args:
            obs_keys (str | List[str]): Column name(s) to retrieve and combine.

        Returns:
            np.ndarray: Integer codes representing the (combined) categories.
                Shape: (n_samples,).
        """
        if isinstance(obs_keys, str):
            obs_keys = [obs_keys]
        labels = None
        for label_key in obs_keys:
            labels_to_str = self.mapped_dataset.get_merged_labels(label_key)
            if labels is None:
                labels = labels_to_str
            else:
                labels = concat_categorical_codes([labels, labels_to_str])
        return np.array(labels.codes)

    def get_unseen_mapped_dataset_elements(self, idx: int):
        """
        Get genes marked as unseen for a specific sample.

        Retrieves the list of genes that were not observed (expression = 0 or
        marked as unseen) for the sample at the given index.

        Args:
            idx (int): Sample index in the dataset.

        Returns:
            List[str]: List of unseen gene identifiers.
        """
        return [str(i)[2:-1] for i in self.mapped_dataset.uns(idx, "unseen_genes")]

    def define_hierarchies(self, clsses: List[str]):
        """
        Define hierarchical label groupings from ontology relationships.

        Uses Bionty to retrieve parent-child relationships for ontology terms,
        then builds groupings mapping parent terms to their descendants.
        Updates encoders to include parent terms and reorders labels so that
        leaf terms (directly predictable) come first.

        Args:
            clsses (List[str]): List of ontology column names to process.

        Raises:
            ValueError: If a class name is not in the supported ontology types.

        Note:
            Modifies self.labels_groupings, self.class_topred, and
            self.mapped_dataset.encoders in place.
        """
        # TODO: use all possible hierarchies instead of just the ones for which we have a sample annotated with
        self.labels_groupings = {}
        self.class_topred = {}
        for clss in clsses:
            if clss not in [
                "cell_type_ontology_term_id",
                "tissue_ontology_term_id",
                "disease_ontology_term_id",
                "development_stage_ontology_term_id",
                "simplified_dev_stage",
                "age_group",
                "assay_ontology_term_id",
                "self_reported_ethnicity_ontology_term_id",
            ]:
                raise ValueError(
                    "class {} not in accepted classes, for now only supported from bionty sources".format(
                        clss
                    )
                )
            elif clss == "cell_type_ontology_term_id":
                parentdf = (
                    bt.CellType.filter()
                    .df(include=["parents__ontology_id", "ontology_id"])
                    .set_index("ontology_id")
                )
            elif clss == "tissue_ontology_term_id":
                parentdf = (
                    bt.Tissue.filter()
                    .df(include=["parents__ontology_id", "ontology_id"])
                    .set_index("ontology_id")
                )
            elif clss == "disease_ontology_term_id":
                parentdf = (
                    bt.Disease.filter()
                    .df(include=["parents__ontology_id", "ontology_id"])
                    .set_index("ontology_id")
                )
            elif clss in [
                "development_stage_ontology_term_id",
                "simplified_dev_stage",
                "age_group",
            ]:
                parentdf = (
                    bt.DevelopmentalStage.filter()
                    .df(include=["parents__ontology_id", "ontology_id"])
                    .set_index("ontology_id")
                )
            elif clss == "assay_ontology_term_id":
                parentdf = (
                    bt.ExperimentalFactor.filter()
                    .df(include=["parents__ontology_id", "ontology_id"])
                    .set_index("ontology_id")
                )
            elif clss == "self_reported_ethnicity_ontology_term_id":
                parentdf = (
                    bt.Ethnicity.filter()
                    .df(include=["parents__ontology_id", "ontology_id"])
                    .set_index("ontology_id")
                )

            else:
                raise ValueError(
                    "class {} not in accepted classes, for now only supported from bionty sources".format(
                        clss
                    )
                )
            cats = set(self.mapped_dataset.encoders[clss].keys())
            groupings, _, leaf_labels = get_ancestry_mapping(cats, parentdf)
            groupings.pop(None, None)
            for i, j in groupings.items():
                if len(j) == 0:
                    # that should not happen
                    import pdb

                    pdb.set_trace()
                    groupings.pop(i)

            self.labels_groupings[clss] = groupings
            if clss in self.clss_to_predict:
                # if we have added new clss, we need to update the encoder with them too.
                mlength = len(self.mapped_dataset.encoders[clss])

                mlength -= (
                    1
                    if self.mapped_dataset.unknown_label
                    in self.mapped_dataset.encoders[clss].keys()
                    else 0
                )

                for i, v in enumerate(
                    set(groupings.keys())
                    - set(self.mapped_dataset.encoders[clss].keys())
                ):
                    self.mapped_dataset.encoders[clss].update({v: mlength + i})

                # we need to change the ordering so that the things that can't be predicted appear afterward
                self.class_topred[clss] = leaf_labels
                c = 0
                update = {}
                mlength = len(leaf_labels)
                mlength -= (
                    1
                    if self.mapped_dataset.unknown_label
                    in self.mapped_dataset.encoders[clss].keys()
                    else 0
                )
                for k, v in self.mapped_dataset.encoders[clss].items():
                    if k in self.labels_groupings[clss].keys():
                        update.update({k: mlength + c})
                        c += 1
                    elif k == self.mapped_dataset.unknown_label:
                        update.update({k: v})
                        self.class_topred[clss] -= set([k])
                    else:
                        update.update({k: v - c})
                self.mapped_dataset.encoders[clss] = update


class SimpleAnnDataset(torchDataset):

    def __init__(
        self,
        adata: AnnData,
        obs_to_output: Optional[List[str]] = [],
        layer: Optional[str] = None,
        get_knn_cells: bool = False,
        encoder: Optional[dict[str, dict]] = None,
    ):
        """
        Simple PyTorch Dataset wrapper for a single AnnData object.

        Provides a lightweight interface for using AnnData with PyTorch DataLoaders,
        compatible with the scDataLoader collator. Useful for inference on new data
        that isn't stored in LaminDB.

        Args:
            adata (AnnData): AnnData object containing expression data.
            obs_to_output (List[str], optional): Observation columns to include in
                output dictionaries. Defaults to [].
            layer (str, optional): Layer name to use for expression values. If None,
                uses adata.X. Defaults to None.
            get_knn_cells (bool, optional): Whether to include k-nearest neighbor
                expression data. Requires precomputed neighbors in adata.obsp.
                Defaults to False.
            encoder (dict[str, dict], optional): Dictionary mapping observation column
                names to encoding dictionaries (str -> int). Defaults to None.

        Attributes:
            adataX (np.ndarray): Dense expression matrix.
            encoder (dict): Label encoders.
            obs_to_output (pd.DataFrame): Observation metadata to include.
            distances (scipy.sparse matrix): KNN distance matrix (if get_knn_cells=True).

        Raises:
            ValueError: If get_knn_cells=True but "connectivities" not in adata.obsp.

        Example:
            >>> dataset = SimpleAnnDataset(
            ...     adata=my_adata,
            ...     obs_to_output=["cell_type", "organism_ontology_term_id"],
            ...     encoder={"organism_ontology_term_id": {"NCBITaxon:9606": 0}},
            ... )
            >>> loader = DataLoader(dataset, batch_size=32, collate_fn=collator)
        """
        self.adataX = adata.layers[layer] if layer is not None else adata.X
        self.adataX = self.adataX.toarray() if issparse(self.adataX) else self.adataX
        self.encoder = encoder if encoder is not None else {}

        self.obs_to_output = adata.obs[obs_to_output]
        self.get_knn_cells = get_knn_cells
        if get_knn_cells and "connectivities" not in adata.obsp:
            raise ValueError("neighbors key not found in adata.obsm")
        if get_knn_cells:
            self.distances = adata.obsp["distances"]

    def __len__(self):
        return self.adataX.shape[0]

    def __iter__(self):
        for idx in range(self.adataX.shape[0]):
            out = self.__getitem__(idx)
            yield out

    def __getitem__(self, idx):
        out = {"X": self.adataX[idx].reshape(-1)}
        # put the observation into the output and encode if needed
        for name, val in self.obs_to_output.iloc[idx].items():
            out.update({name: self.encoder[name][val] if name in self.encoder else val})
        if self.get_knn_cells:
            distances = self.distances[idx].toarray()[0]
            nn_idx = np.argsort(-1 / (distances - 1e-6))[:6]
            out["knn_cells"] = np.array(
                [self.adataX[i].reshape(-1) for i in nn_idx],
                dtype=int,
            )
            out["knn_cells_info"] = distances[nn_idx]
        return out


def mapped(
    dataset,
    obs_keys: List[str] | None = None,
    obsm_keys: List[str] | None = None,
    obs_filter: dict[str, str | tuple[str, ...]] | None = None,
    join: Literal["inner", "outer"] | None = "inner",
    encode_labels: bool | List[str] = True,
    unknown_label: str | dict[str, str] | None = None,
    cache_categories: bool = True,
    parallel: bool = False,
    dtype: str | None = None,
    stream: bool = False,
    is_run_input: bool | None = None,
    metacell_mode: bool = False,
    meta_assays: List[str] = ["EFO:0022857", "EFO:0010961"],
    get_knn_cells: bool = False,
    store_location: str | None = None,
    force_recompute_indices: bool = False,
) -> MappedCollection:
    """
    Create a MappedCollection from a LaminDB Collection.

    Factory function that handles artifact path resolution (staging or streaming)
    and creates a MappedCollection for efficient access to multiple h5ad/zarr files.

    Args:
        dataset (ln.Collection): LaminDB Collection containing artifacts to map.
        obs_keys (List[str], optional): Observation columns to load. Defaults to None.
        obsm_keys (List[str], optional): Obsm keys to load. Defaults to None.
        obs_filter (dict, optional): Filter observations by column values.
            Keys are column names, values are allowed values. Defaults to None.
        join (str, optional): How to join variables across files. "inner" for
            intersection, "outer" for union. Defaults to "inner".
        encode_labels (bool | List[str], optional): Whether/which columns to
            integer-encode. True encodes all obs_keys. Defaults to True.
        unknown_label (str | dict, optional): Label to use for unknown/missing
            categories. Defaults to None.
        cache_categories (bool, optional): Whether to cache category mappings.
            Defaults to True.
        parallel (bool, optional): Enable parallel data loading. Defaults to False.
        dtype (str, optional): Data type for expression values. Defaults to None.
        stream (bool, optional): If True, stream from cloud storage instead of
            staging locally. Defaults to False.
        is_run_input (bool, optional): Track as run input in LaminDB. Defaults to None.
        metacell_mode (bool, optional): Enable metacell aggregation. Defaults to False.
        meta_assays (List[str], optional): Assay types to treat as metacell-like.
            Defaults to ["EFO:0022857", "EFO:0010961"].
        get_knn_cells (bool, optional): Include KNN neighbor data. Defaults to False.
        store_location (str, optional): Cache directory path. Defaults to None.
        force_recompute_indices (bool, optional): Force recompute cached data.
            Defaults to False.

    Returns:
        MappedCollection: Mapped collection for data access.

    Note:
        Artifacts with suffixes other than .h5ad, .zrad, .zarr are ignored.
        Non-existent paths are skipped with a warning.
    """
    path_list = []
    for artifact in dataset.artifacts.all():
        if artifact.suffix not in {".h5ad", ".zrad", ".zarr"}:
            print(f"Ignoring artifact with suffix {artifact.suffix}")
            continue
        elif not artifact.path.exists():
            print(f"Path does not exist for artifact with suffix {artifact.suffix}")
            continue
        elif not stream:
            path_list.append(artifact.stage())
        else:
            path_list.append(artifact.path)
    ds = MappedCollection(
        path_list=path_list,
        obs_keys=obs_keys,
        obsm_keys=obsm_keys,
        obs_filter=obs_filter,
        join=join,
        encode_labels=encode_labels,
        unknown_label=unknown_label,
        cache_categories=cache_categories,
        parallel=parallel,
        dtype=dtype,
        meta_assays=meta_assays,
        metacell_mode=metacell_mode,
        get_knn_cells=get_knn_cells,
        store_location=store_location,
        force_recompute_indices=force_recompute_indices,
    )
    return ds


def concat_categorical_codes(series_list: List[pd.Categorical]) -> pd.Categorical:
    """
    Efficiently combine multiple categorical arrays into a single encoding.

    Creates a combined categorical where each unique combination of input
    categories gets a unique code. Only combinations that exist in the data
    are assigned codes (sparse encoding).

    Args:
        series_list (List[pd.Categorical]): List of categorical arrays to combine.
            All arrays must have the same length.

    Returns:
        pd.Categorical: Combined categorical with compressed codes representing
            unique combinations present in the data.

    Example:
        >>> cat1 = pd.Categorical(["a", "a", "b", "b"])
        >>> cat2 = pd.Categorical(["x", "y", "x", "y"])
        >>> combined = concat_categorical_codes([cat1, cat2])
        >>> # Results in 4 unique codes for (a,x), (a,y), (b,x), (b,y)
    """
    # Get the codes for each categorical
    codes_list = [s.codes.astype(np.int32) for s in series_list]
    n_cats = [len(s.categories) for s in series_list]

    # Calculate combined codes
    combined_codes = codes_list[0]
    multiplier = n_cats[0]
    for codes, n_cat in zip(codes_list[1:], n_cats[1:]):
        combined_codes = (combined_codes * n_cat) + codes
        multiplier *= n_cat

    # Find unique combinations that actually exist in the data
    unique_existing_codes = np.unique(combined_codes)

    # Create a mapping from old codes to new compressed codes
    code_mapping = {old: new for new, old in enumerate(unique_existing_codes)}

    # Map the combined codes to their new compressed values
    combined_codes = np.array([code_mapping[code] for code in combined_codes])

    # Create final categorical with only existing combinations
    return pd.Categorical.from_codes(
        codes=combined_codes,
        categories=np.arange(len(unique_existing_codes)),
        ordered=False,
    )
