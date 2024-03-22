from dataclasses import dataclass, field

import lamindb as ln
import bionty as bt
import pandas as pd
from torch.utils.data import Dataset as torchDataset
from typing import Union, Optional, Literal
from scdataloader import mapped
import warnings

from anndata import AnnData

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
            #"dpt_group",
            #"heat_diff",
            #"nnz",
        ]
    )
    # set of obs to prepare for prediction (encode)
    clss_to_pred: Optional[list[str]] = field(default_factory=list)
    # set of obs that need to be hierarchically prepared
    hierarchical_clss: Optional[list[str]] = field(default_factory=list)
    join_vars: Optional[Literal["auto", "inner", "None"]] = "None"

    def __post_init__(self):
        self.mapped_dataset = mapped.mapped(
            self.lamin_dataset,
            label_keys=self.obs,
            encode_labels=self.clss_to_pred,
            stream=True,
            parallel=True,
            join_vars=self.join_vars,
        )
        print(
            "won't do any check but we recommend to have your dataset coming from local storage"
        )
        self.class_groupings = {}
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
                    update = {}
                    c = 0
                    for k, v in self.mapped_dataset.encoders[clss].items():
                        if k == self.mapped_dataset.unknown_class:
                            update.update({k: v})
                            c += 1
                            self.class_topred[clss] -= set([k])
                        else:
                            update.update({k: v - c})
                    self.mapped_dataset.encoders[clss] = update

        if self.genedf is None:
            self.genedf = load_genes(self.organisms)

        self.genedf.columns = self.genedf.columns.astype(str)
        for organism in self.organisms:
            ogenedf = self.genedf[self.genedf.organism == organism]
            self.mapped_dataset._check_aligned_vars(ogenedf.index.tolist())

    def __len__(self, **kwargs):
        return self.mapped_dataset.__len__(**kwargs)

    @property
    def encoder(self):
        return self.mapped_dataset.encoders

    def __getitem__(self, *args, **kwargs):
        item = self.mapped_dataset.__getitem__(*args, **kwargs)
        # import pdb

        # pdb.set_trace()
        # item.update(
        #    {"unseen_genes": self.get_unseen_mapped_dataset_elements(*args, **kwargs)}
        # )
        # ret = {}
        # ret["count"] = item[0]
        # for i, val in enumerate(self.obs):
        #    ret[val] = item[1][i]
        ## mark unseen genes with a flag
        ## send the associated
        # print(item[0].shape)
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
            + "     {} join_vars\n".format(len(self.join_vars))
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
        return self.mapped_dataset.get_label_weights(*args, **kwargs)

    def get_unseen_mapped_dataset_elements(self, idx: int):
        return [str(i)[2:-1] for i in self.mapped_dataset.uns(idx, "unseen_genes")]

    def define_hierarchies(self, labels: list[str]):
        # TODO: use all possible hierarchies instead of just the ones for which we have a sample annotated with
        self.class_groupings = {}
        self.class_topred = {}
        for label in labels:
            if label not in [
                "cell_type_ontology_term_id",
                "tissue_ontology_term_id",
                "disease_ontology_term_id",
                "development_stage_ontology_term_id",
                "assay_ontology_term_id",
                "self_reported_ethnicity_ontology_term_id",
            ]:
                raise ValueError(
                    "label {} not in accepted labels, for now only supported from bionty sources".format(
                        label
                    )
                )
            elif label == "cell_type_ontology_term_id":
                parentdf = (
                    bt.CellType.filter()
                    .df(include=["parents__ontology_id"])
                    .set_index("ontology_id")
                )
            elif label == "tissue_ontology_term_id":
                parentdf = (
                    bt.Tissue.filter()
                    .df(include=["parents__ontology_id"])
                    .set_index("ontology_id")
                )
            elif label == "disease_ontology_term_id":
                parentdf = (
                    bt.Disease.filter()
                    .df(include=["parents__ontology_id"])
                    .set_index("ontology_id")
                )
            elif label == "development_stage_ontology_term_id":
                parentdf = (
                    bt.DevelopmentalStage.filter()
                    .df(include=["parents__ontology_id"])
                    .set_index("ontology_id")
                )
            elif label == "assay_ontology_term_id":
                parentdf = (
                    bt.ExperimentalFactor.filter()
                    .df(include=["parents__ontology_id"])
                    .set_index("ontology_id")
                )
            elif label == "self_reported_ethnicity_ontology_term_id":
                parentdf = (
                    bt.Ethnicity.filter()
                    .df(include=["parents__ontology_id"])
                    .set_index("ontology_id")
                )

            else:
                raise ValueError(
                    "label {} not in accepted labels, for now only supported from bionty sources".format(
                        label
                    )
                )
            cats = self.mapped_dataset.get_merged_categories(label)
            addition = set(LABELS_TOADD.get(label, {}).values())
            cats |= addition
            # import pdb

            # pdb.set_trace()
            groupings, _, lclass = get_ancestry_mapping(cats, parentdf)
            for i, j in groupings.items():
                if len(j) == 0:
                    groupings.pop(i)
            self.class_groupings[label] = groupings
            if label in self.clss_to_pred:
                # if we have added new labels, we need to update the encoder with them too.
                mlength = len(self.mapped_dataset.encoders[label])
                mlength -= (
                    1
                    if self.mapped_dataset.unknown_class
                    in self.mapped_dataset.encoders[label].keys()
                    else 0
                )

                for i, v in enumerate(
                    addition - set(self.mapped_dataset.encoders[label].keys())
                ):
                    self.mapped_dataset.encoders[label].update({v: mlength + i})
                # we need to change the ordering so that the things that can't be predicted appear afterward

                self.class_topred[label] = lclass
                c = 0
                d = 0
                update = {}
                mlength = len(lclass)
                # import pdb

                # pdb.set_trace()
                mlength -= (
                    1
                    if self.mapped_dataset.unknown_class
                    in self.mapped_dataset.encoders[label].keys()
                    else 0
                )
                for k, v in self.mapped_dataset.encoders[label].items():
                    if k in self.class_groupings[label].keys():
                        update.update({k: mlength + c})
                        c += 1
                    elif k == self.mapped_dataset.unknown_class:
                        update.update({k: v})
                        d += 1
                        self.class_topred[label] -= set([k])
                    else:
                        update.update({k: (v - c) - d})
                self.mapped_dataset.encoders[label] = update


class SimpleAnnDataset:
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
        self.adata = adata
        self.obs_to_output = obs_to_output
        self.layer = layer

    def __len__(self):
        return self.adata.shape[0]

    def __getitem__(self, idx):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            if self.layer is not None:
                out = {"x": self.adata.layers[self.layer][idx].toarray().reshape(-1)}
            else:
                out = {"x": self.adata.X[idx].toarray().reshape(-1)}
            for i in self.obs_to_output:
                out.update({i: self.adata.obs.iloc[idx][i]})
        return out
