from dataclasses import dataclass, field

import lamindb as ln

# ln.connect("scprint")

import bionty as bt
import pandas as pd
from torch.utils.data import Dataset as torchDataset
from typing import Union, Optional, Literal
from scdataloader.mapped import MappedCollection
import warnings

from anndata import AnnData
from scipy.sparse import issparse

from scdataloader.utils import get_ancestry_mapping, load_genes

from .config import LABELS_TOADD


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
        organisms (list[str]): list of organisms to load
            (for now only validates the the genes map to this organism)
        obs (list[str]): list of observations to load from the Collection
        clss_to_pred (list[str]): list of observations to encode
        join_vars (flag): join variables @see :meth:`~lamindb.Dataset.mapped`.
        hierarchical_clss: list of observations to map to a hierarchy using lamin's bionty
    """

    lamin_dataset: ln.Collection
    genedf: Optional[pd.DataFrame] = None
    organisms: Optional[Union[list[str], str]] = field(
        default_factory=["NCBITaxon:9606", "NCBITaxon:10090"]
    )
    obs: Optional[list[str]] = field(
        default_factory=[
            "self_reported_ethnicity_ontology_term_id",
            "assay_ontology_term_id",
            "development_stage_ontology_term_id",
            "disease_ontology_term_id",
            "cell_type_ontology_term_id",
            "tissue_ontology_term_id",
            "sex_ontology_term_id",
            #'dataset_id',
            #'cell_culture',
            # "dpt_group",
            # "heat_diff",
            # "nnz",
        ]
    )
    # set of obs to prepare for prediction (encode)
    clss_to_pred: Optional[list[str]] = field(default_factory=list)
    # set of obs that need to be hierarchically prepared
    hierarchical_clss: Optional[list[str]] = field(default_factory=list)
    join_vars: Literal["inner", "outer"] | None = None

    def __post_init__(self):
        self.mapped_dataset = mapped(
            self.lamin_dataset,
            obs_keys=self.obs,
            join=self.join_vars,
            encode_labels=self.clss_to_pred,
            unknown_label="unknown",
            stream=True,
            parallel=True,
        )
        print(
            "won't do any check but we recommend to have your dataset coming from local storage"
        )
        self.labels_groupings = {}
        self.class_topred = {}
        # generate tree from ontologies
        if len(self.hierarchical_clss) > 0:
            self.define_hierarchies(self.hierarchical_clss)
        if len(self.clss_to_pred) > 0:
            for clss in self.clss_to_pred:
                if clss not in self.hierarchical_clss:
                    # otherwise it's already been done
                    self.class_topred[clss] = self.mapped_dataset.get_merged_categories(
                        clss
                    )
                    if (
                        self.mapped_dataset.unknown_label
                        in self.mapped_dataset.encoders[clss].keys()
                    ):
                        self.class_topred[clss] -= set(
                            [self.mapped_dataset.unknown_label]
                        )

        if self.genedf is None:
            self.genedf = load_genes(self.organisms)

        self.genedf.columns = self.genedf.columns.astype(str)
        self.mapped_dataset._check_aligned_vars(self.genedf.index.tolist())

    def __len__(self, **kwargs):
        return self.mapped_dataset.__len__(**kwargs)

    @property
    def encoder(self):
        return self.mapped_dataset.encoders

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
            + "     {} labels\n".format(len(self.obs))
            + "     {} clss_to_pred\n".format(len(self.clss_to_pred))
            + "     {} hierarchical_clss\n".format(len(self.hierarchical_clss))
            + "     {} organisms\n".format(len(self.organisms))
            + (
                "dataset contains {} classes to predict\n".format(
                    sum([len(self.class_topred[i]) for i in self.class_topred])
                )
                if len(self.class_topred) > 0
                else ""
            )
        )

    def get_label_weights(self, *args, **kwargs):
        """
        get_label_weights is a wrapper around mappedDataset.get_label_weights

        Returns:
            dict: dictionary of weights for each label
        """
        return self.mapped_dataset.get_label_weights(*args, **kwargs)

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
            elif clss == "development_stage_ontology_term_id":
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
            cats = self.mapped_dataset.get_merged_categories(clss)
            addition = set(LABELS_TOADD.get(clss, {}).values())
            cats |= addition
            groupings, _, leaf_labels = get_ancestry_mapping(cats, parentdf)
            for i, j in groupings.items():
                if len(j) == 0:
                    groupings.pop(i)
            self.labels_groupings[clss] = groupings
            if clss in self.clss_to_pred:
                # if we have added new clss, we need to update the encoder with them too.
                mlength = len(self.mapped_dataset.encoders[clss])

                mlength -= (
                    1
                    if self.mapped_dataset.unknown_label
                    in self.mapped_dataset.encoders[clss].keys()
                    else 0
                )

                for i, v in enumerate(
                    addition - set(self.mapped_dataset.encoders[clss].keys())
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
    ):
        """
        SimpleAnnDataset is a simple dataloader for an AnnData dataset. this is to interface nicely with the rest of
        scDataloader and with your model during inference.

        Args:
        ----
            adata (anndata.AnnData): anndata object to use
            obs_to_output (list[str]): list of observations to output from anndata.obs
            layer (str): layer of the anndata to use
        """
        self.adataX = adata.layers[layer] if layer is not None else adata.X
        self.adataX = self.adataX.toarray() if issparse(self.adataX) else self.adataX
        self.obs_to_output = adata.obs[obs_to_output]

    def __len__(self):
        return self.adataX.shape[0]

    def __iter__(self):
        for idx, obs in enumerate(self.adata.obs.itertuples(index=False)):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                out = {"X": self.adataX[idx].reshape(-1)}
                out.update(
                    {name: val for name, val in self.obs_to_output.iloc[idx].items()}
                )
                yield out

    def __getitem__(self, idx):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            out = {"X": self.adataX[idx].reshape(-1)}
            out.update(
                {name: val for name, val in self.obs_to_output.iloc[idx].items()}
            )
        return out


def mapped(
    dataset,
    obs_keys: list[str] | None = None,
    join: Literal["inner", "outer"] | None = "inner",
    encode_labels: bool | list[str] = True,
    unknown_label: str | dict[str, str] | None = None,
    cache_categories: bool = True,
    parallel: bool = False,
    dtype: str | None = None,
    stream: bool = False,
    is_run_input: bool | None = None,
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
        join=join,
        encode_labels=encode_labels,
        unknown_label=unknown_label,
        cache_categories=cache_categories,
        parallel=parallel,
        dtype=dtype,
    )
    return ds
