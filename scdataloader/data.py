from dataclasses import dataclass, field

import lamindb as ln
import lnschema_bionty as lb
import pandas as pd
from torch.utils.data import Dataset as torchDataset
from typing import Union
from scdataloader import mapped
import warnings

# TODO: manage load gene embeddings to make
# from scprint.dataloader.embedder import embed
from scdataloader.utils import get_ancestry_mapping, load_genes

LABELS_TOADD = {
    "assay_ontology_term_id": {
        "10x transcription profiling": "EFO:0030003",
        "spatial transcriptomics": "EFO:0008994",
        "10x 3' transcription profiling": "EFO:0030003",
        "10x 5' transcription profiling": "EFO:0030004",
    },
    "disease_ontology_term_id": {
        "metabolic disease": "MONDO:0005066",
        "chronic kidney disease": "MONDO:0005300",
        "chromosomal disorder": "MONDO:0019040",
        "infectious disease": "MONDO:0005550",
        "inflammatory disease": "MONDO:0021166",
        # "immune system disease",
        "disorder of development or morphogenesis": "MONDO:0021147",
        "mitochondrial disease": "MONDO:0044970",
        "psychiatric disorder": "MONDO:0002025",
        "cancer or benign tumor": "MONDO:0002025",
        "neoplasm": "MONDO:0005070",
    },
    "cell_type_ontology_term_id": {
        "progenitor cell": "CL:0011026",
        "hematopoietic cell": "CL:0000988",
        "myoblast": "CL:0000056",
        "myeloid cell": "CL:0000763",
        "neuron": "CL:0000540",
        "electrically active cell": "CL:0000211",
        "epithelial cell": "CL:0000066",
        "secretory cell": "CL:0000151",
        "stem cell": "CL:0000034",
        "non-terminally differentiated cell": "CL:0000055",
        "supporting cell": "CL:0000630",
    },
}


@dataclass
class Dataset(torchDataset):
    """
    Dataset class to load a bunch of anndata from a lamin dataset in a memory efficient way.

    For an example, see :meth:`~lamindb.Dataset.mapped`.

    .. note::

        A similar data loader exists `here
        <https://github.com/Genentech/scimilarity>`__.

    Attributes:
    ----
        lamin_dataset (lamindb.Dataset): lamin dataset to load
        genedf (pd.Dataframe): dataframe containing the genes to load
        gene_embedding: dataframe containing the gene embeddings
        organisms (list[str]): list of organisms to load
        obs (list[str]): list of observations to load
        clss_to_pred (list[str]): list of observations to encode
        hierarchical_clss: list of observations to map to a hierarchy
    """

    lamin_dataset: ln.Collection
    genedf: pd.DataFrame = None
    # gene_embedding: pd.DataFrame = None  # TODO: make it part of specialized dataset
    organisms: Union[list[str], str] = field(
        default_factory=["NCBITaxon:9606", "NCBITaxon:10090"]
    )
    obs: list[str] = field(
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
            "dpt_group",
            "heat_diff",
            "nnz",
        ]
    )
    # set of obs to prepare for prediction (encode)
    clss_to_pred: list[str] = field(default_factory=list)
    # set of obs that need to be hierarchically prepared
    hierarchical_clss: list[str] = field(default_factory=list)
    join_vars: str = "None"

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
        #item.update({"unseen_genes": self.get_unseen_mapped_dataset_elements(*args, **kwargs)})
        # ret = {}
        # ret["count"] = item[0]
        # for i, val in enumerate(self.obs):
        #    ret[val] = item[1][i]
        ## mark unseen genes with a flag
        ## send the associated
        # print(item[0].shape)
        return item

    def __repr__(self):
        print(
            "total dataset size is {} Gb".format(
                sum([file.size for file in self.lamin_dataset.artifacts.all()]) / 1e9
            )
        )
        print("---")
        print("dataset contains:")
        print("     {} cells".format(self.mapped_dataset.__len__()))
        print("     {} genes".format(self.genedf.shape[0]))
        print("     {} labels".format(len(self.obs)))
        print("     {} organisms".format(len(self.organisms)))
        print(
            "dataset contains {} classes to predict".format(
                sum([len(self.class_topred[i]) for i in self.class_topred])
            )
        )
        # print("embedding size is {}".format(self.gene_embedding.shape[1]))
        return ""

    def get_label_weights(self, *args, **kwargs):
        return self.mapped_dataset.get_label_weights(*args, **kwargs)

    def get_unseen_mapped_dataset_elements(self, idx):
        return [str(i)[2:-1] for i in self.mapped_dataset.uns(idx, "unseen_genes")]

    # def load_embeddings(self, genedfs, embedding_size=128, cache=True):
    #    embeddings = []
    #    for o in self.organisms:
    #        genedf = genedfs[genedfs.organism == o]
    #        org_name = lb.Organism.filter(ontology_id=o).one().scientific_name
    #        embedding = embed(
    #            genedf=genedf,
    #            organism=org_name,
    #            cache=cache,
    #            fasta_path="/tmp/data/fasta/",
    #            embedding_size=embedding_size,
    #        )
    #        genedf = pd.concat(
    #            [genedf.set_index("ensembl_gene_id"), embedding], axis=1, join="inner"
    #        )
    #        genedf.columns = genedf.columns.astype(str)
    #        embeddings.append(genedf)
    #    return pd.concat(embeddings)

    def define_hierarchies(self, labels):
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
                    lb.CellType.filter()
                    .df(include=["parents__ontology_id"])
                    .set_index("ontology_id")
                )
            elif label == "tissue_ontology_term_id":
                parentdf = (
                    lb.Tissue.filter()
                    .df(include=["parents__ontology_id"])
                    .set_index("ontology_id")
                )
            elif label == "disease_ontology_term_id":
                parentdf = (
                    lb.Disease.filter()
                    .df(include=["parents__ontology_id"])
                    .set_index("ontology_id")
                )
            elif label == "development_stage_ontology_term_id":
                parentdf = (
                    lb.DevelopmentalStage.filter()
                    .df(include=["parents__ontology_id"])
                    .set_index("ontology_id")
                )
            elif label == "assay_ontology_term_id":
                parentdf = (
                    lb.ExperimentalFactor.filter()
                    .df(include=["parents__ontology_id"])
                    .set_index("ontology_id")
                )
            elif label == "self_reported_ethnicity_ontology_term_id":
                parentdf = (
                    lb.Ethnicity.filter()
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
    def __init__(self, adata, obs_to_output=[], layer=None):
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
