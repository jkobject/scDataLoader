import warnings
from collections import Counter
from dataclasses import dataclass, field
from functools import reduce
from typing import Literal, Optional, Union

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
    Dataset class to load a bunch of anndata from a lamin dataset (Collection) in a memory efficient way.

    This serves as a wrapper around lamin's mappedCollection to provide more features,
    mostly, the management of hierarchical labels, the encoding of labels, the management of multiple species

    For an example of mappedDataset, see :meth:`~lamindb.Dataset.mapped`.

    .. note::

        A related data loader exists `here
        <https://github.com/Genentech/scimilarity>`__.

    Args:
    ----
        lamin_dataset (lamindb.Dataset): lamin dataset to load
        genedf (pd.Dataframe): dataframe containing the genes to load
        obs (list[str]): list of observations to load from the Collection
        clss_to_predict (list[str]): list of observations to encode
        join_vars (flag): join variables @see :meth:`~lamindb.Dataset.mapped`.
        hierarchical_clss: list of observations to map to a hierarchy using lamin's bionty
        metacell_mode (float, optional): The mode to use for metacell sampling. Defaults to 0.0.
        get_knn_cells (bool, optional): Whether to get the k-nearest neighbors of each cell. Defaults to False.
        store_location (str, optional): The location to store the sampler indices. Defaults to None.
        force_recompute_indices (bool, optional): Whether to force recompute the sampler indices. Defaults to False.
    """

    lamin_dataset: ln.Collection
    genedf: Optional[pd.DataFrame] = None
    # set of obs to prepare for prediction (encode)
    clss_to_predict: Optional[list[str]] = field(default_factory=list)
    # set of obs that need to be hierarchically prepared
    hierarchical_clss: Optional[list[str]] = field(default_factory=list)
    join_vars: Literal["inner", "outer"] | None = None
    metacell_mode: float = 0.0
    get_knn_cells: bool = False
    store_location: str | None = None
    force_recompute_indices: bool = False

    def __post_init__(self):
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
            self.organisms.sort()
            self.genedf = load_genes(self.organisms)
        else:
            self.organisms = None

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

    def get_label_weights(
        self,
        obs_keys: str | list[str],
        scaler: int = 10,
        return_categories=False,
    ):
        """Get all weights for the given label keys."""
        if isinstance(obs_keys, str):
            obs_keys = [obs_keys]
        labels = None
        for label_key in obs_keys:
            labels_to_str = self.mapped_dataset.get_merged_labels(label_key)
            if labels is None:
                labels = labels_to_str
            else:
                labels = concat_categorical_codes([labels, labels_to_str])
        counter = Counter(labels.codes)  # type: ignore
        if return_categories:
            counter = np.array(list(counter.values()))
            weights = scaler / (counter + scaler)
            return weights, np.array(labels.codes)
        else:
            counts = np.array([counter[label] for label in labels.codes])
            if scaler is None:
                weights = 1.0 / counts
            else:
                weights = scaler / (counts + scaler)
            return weights

    def get_unseen_mapped_dataset_elements(self, idx: int):
        """
        get_unseen_mapped_dataset_elements is a wrapper around mappedDataset.get_unseen_mapped_dataset_elements

        Args:
            idx (int): index of the element to get

        Returns:
            list[str]: list of unseen genes
        """
        return [str(i)[2:-1] for i in self.mapped_dataset.uns(idx, "unseen_genes")]

    def define_hierarchies(self, clsses: list[str]):
        """
        define_hierarchies is a method to define the hierarchies for the classes to predict

        Args:
            clsses (list[str]): list of classes to predict

        Raises:
            ValueError: if the class is not in the accepted classes
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
                    .df(include=["parents__ontology_id"])
                    .set_index("ontology_id")
                )
            elif clss == "tissue_ontology_term_id":
                parentdf = (
                    bt.Tissue.filter()
                    .df(include=["parents__ontology_id"])
                    .set_index("ontology_id")
                )
            elif clss == "disease_ontology_term_id":
                parentdf = (
                    bt.Disease.filter()
                    .df(include=["parents__ontology_id"])
                    .set_index("ontology_id")
                )
            elif clss in [
                "development_stage_ontology_term_id",
                "simplified_dev_stage",
                "age_group",
            ]:
                parentdf = (
                    bt.DevelopmentalStage.filter()
                    .df(include=["parents__ontology_id"])
                    .set_index("ontology_id")
                )
            elif clss == "assay_ontology_term_id":
                parentdf = (
                    bt.ExperimentalFactor.filter()
                    .df(include=["parents__ontology_id"])
                    .set_index("ontology_id")
                )
            elif clss == "self_reported_ethnicity_ontology_term_id":
                parentdf = (
                    bt.Ethnicity.filter()
                    .df(include=["parents__ontology_id"])
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
        obs_to_output: Optional[list[str]] = [],
        layer: Optional[str] = None,
        get_knn_cells: bool = False,
    ):
        """
        SimpleAnnDataset is a simple dataloader for an AnnData dataset. this is to interface nicely with the rest of
        scDataloader and with your model during inference.

        Args:
        ----
            adata (anndata.AnnData): anndata object to use
            obs_to_output (list[str]): list of observations to output from anndata.obs
            layer (str): layer of the anndata to use
            get_knn_cells (bool): whether to get the knn cells
        """
        self.adataX = adata.layers[layer] if layer is not None else adata.X
        self.adataX = self.adataX.toarray() if issparse(self.adataX) else self.adataX

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
            out = {"X": self.adataX[idx].reshape(-1)}
            out.update(
                {name: val for name, val in self.obs_to_output.iloc[idx].items()}
            )
            if self.get_knn_cells:
                distances = self.distances[idx].toarray()[0]
                nn_idx = np.argsort(-1 / (distances - 1e-6))[:6]
                out["knn_cells"] = np.array(
                    [self.adataX[i].reshape(-1) for i in nn_idx],
                    dtype=int,
                )
                out["distances"] = distances[nn_idx]
            yield out

    def __getitem__(self, idx):
        out = {"X": self.adataX[idx].reshape(-1)}
        out.update({name: val for name, val in self.obs_to_output.iloc[idx].items()})
        if self.get_knn_cells:
            distances = self.distances[idx].toarray()[0]
            nn_idx = np.argsort(-1 / (distances - 1e-6))[:6]
            out["knn_cells"] = np.array(
                [self.adataX[i].reshape(-1) for i in nn_idx],
                dtype=int,
            )
            out["distances"] = distances[nn_idx]
        return out


def mapped(
    dataset,
    obs_keys: list[str] | None = None,
    obsm_keys: list[str] | None = None,
    obs_filter: dict[str, str | tuple[str, ...]] | None = None,
    join: Literal["inner", "outer"] | None = "inner",
    encode_labels: bool | list[str] = True,
    unknown_label: str | dict[str, str] | None = None,
    cache_categories: bool = True,
    parallel: bool = False,
    dtype: str | None = None,
    stream: bool = False,
    is_run_input: bool | None = None,
    metacell_mode: bool = False,
    meta_assays: list[str] = ["EFO:0022857", "EFO:0010961"],
    get_knn_cells: bool = False,
    store_location: str | None = None,
    force_recompute_indices: bool = False,
) -> MappedCollection:
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


def concat_categorical_codes(series_list: list[pd.Categorical]) -> pd.Categorical:
    """Efficiently combine multiple categorical data using their codes,
    only creating categories for combinations that exist in the data.

    Args:
        series_list: List of pandas Categorical data

    Returns:
        Combined Categorical with only existing combinations
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
