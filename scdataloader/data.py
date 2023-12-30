import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from torch.utils.data import Dataset as torchDataset

import lamindb as ln
import lnschema_bionty as lb

# TODO: manage load gene embeddings to make
# from scprint.dataloader.embedder import embed
from scdataloader.utils import get_ancestry_mapping, pd_load_cached
from scdataloader import mapped


LABELS_TOADD = {
    "assay_ontology_term_id": [
        "10x transcription profiling",
        "spatial transcriptomics",
        "10x 3' transcription profiling",
        "10x 5' transcription profiling",
    ],
    "disease_ontology_term_id": [
        "metabolic disease",
        "chronic kidney disease",
        "chromosomal disorder",
        "infectious disease",
        "inflammatory disease",
        # "immune system disease",
        "disorder of development or morphogenesis",
        "mitochondrial disease",
        "psychiatric disorder",
        "cancer or benign tumor",
        "neoplasm",
    ],
    "cell_type_ontology_term_id": [
        "progenitor cell",
        "hematopoietic cell",
        "myoblast",
        "myeloid cell",
        "neuron",
        "electrically active cell",
        "epithelial cell",
        "secretory cell",
        "stem cell",
        "non-terminally differentiated cell",
        "supporting cell",
    ],
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
        encode_obs (list[str]): list of observations to encode
        map_hierarchy: list of observations to map to a hierarchy
    """

    lamin_dataset: ln.Dataset
    genedf: pd.DataFrame = None
    gene_embedding: pd.DataFrame = None  # TODO: make it part of specialized dataset
    organisms: list[str] = field(default_factory=["NCBITaxon:9606", "NCBITaxon:10090"])
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
    encode_obs: list[str] = field(default_factory=list)
    map_hierarchy: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.mapped_dataset = mapped.mapped(
            self.lamin_dataset,
            label_keys=self.obs,
            encode_labels=self.encode_obs,
            stream=True,
            parallel=True,
            join_vars="None",
        )
        print(
            "won't do any check but we recommend to have your dataset coming from local storage"
        )
        # generate tree from ontologies
        if len(self.map_hierarchy) > 0:
            self.define_hierarchies(self.map_hierarchy)

        if self.genedf is None:
            self.genedf = self.load_genes(self.organisms)

        if self.gene_embedding is None:
            self.gene_embedding = self.load_embeddings(self.genedf)
        else:
            # self.genedf = pd.concat(
            #    [self.genedf.set_index("ensembl_gene_id"), self.gene_embedding],
            #    axis=1,
            #    join="inner",
            # )
            self.genedf.columns = self.genedf.columns.astype(str)

    def __len__(self, **kwargs):
        return self.mapped_dataset.__len__(**kwargs)

    def __getitem__(self, *args, **kwargs):
        item = self.mapped_dataset.__getitem__(*args, **kwargs)
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
        print("embedding size is {}".format(self.gene_embedding.shape[1]))
        return ""

    def get_label_weights(self, *args, **kwargs):
        return self.mapped_dataset.get_label_weights(*args, **kwargs)

    def get_unseen_mapped_dataset_elements(self, idx):
        return [str(i)[2:-1] for i in self.mapped_dataset.uns(idx, "unseen_genes")]

    def use_prior_network(
        self, name="collectri", organism="human", split_complexes=True
    ):
        """
        use_prior_network loads a prior GRN from a list of available networks.

        Args:
            name (str, optional): name of the network to load. Defaults to "collectri".
            organism (str, optional): organism to load the network for. Defaults to "human".
            split_complexes (bool, optional): whether to split complexes into individual genes. Defaults to True.

        Raises:
            ValueError: if the provided name is not amongst the available names.
        """
        # TODO: use omnipath instead
        if name == "tflink":
            TFLINK = "https://cdn.netbiol.org/tflink/download_files/TFLink_Homo_sapiens_interactions_All_simpleFormat_v1.0.tsv.gz"
            net = pd_load_cached(TFLINK)
            net = net.rename(columns={"Name.TF": "regulator", "Name.Target": "target"})
        elif name == "htftarget":
            HTFTARGET = "http://bioinfo.life.hust.edu.cn/static/hTFtarget/file_download/tf-target-infomation.txt"
            net = pd_load_cached(HTFTARGET)
            net = net.rename(columns={"TF": "regulator"})
        elif name == "collectri":
            import decoupler as dc

            net = dc.get_collectri(organism=organism, split_complexes=split_complexes)
            net = net.rename(columns={"source": "regulator"})
        else:
            raise ValueError(
                f"provided name: '{name}' is not amongst the available names."
            )
        self.add_prior_network(net)

    def add_prior_network(self, prior_network: pd.DataFrame):
        # validate the network dataframe
        required_columns: list[str] = ["target", "regulators"]
        optional_columns: list[str] = ["type", "weight"]

        for column in required_columns:
            assert (
                column in prior_network.columns
            ), f"Column '{column}' is missing in the provided network dataframe."

        for column in optional_columns:
            if column not in prior_network.columns:
                print(
                    f"Optional column '{column}' is not present in the provided network dataframe."
                )

        assert (
            prior_network["target"].dtype == "str"
        ), "Column 'target' should be of dtype 'str'."
        assert (
            prior_network["regulators"].dtype == "str"
        ), "Column 'regulators' should be of dtype 'str'."

        if "type" in prior_network.columns:
            assert (
                prior_network["type"].dtype == "str"
            ), "Column 'type' should be of dtype 'str'."

        if "weight" in prior_network.columns:
            assert (
                prior_network["weight"].dtype == "float"
            ), "Column 'weight' should be of dtype 'float'."

        # TODO: check that we match the genes in the network to the genes in the dataset

        print(
            "loaded {:.2f}% of the edges".format((len(prior_network) / init_len) * 100)
        )
        # TODO: transform it into a sparse matrix
        self.prior_network = prior_network
        self.network_size = len(prior_network)
        # self.overlap =
        # self.edge_freq

    def load_genes(self, organisms):
        organismdf = []
        for o in organisms:
            organism = lb.Gene(organism=lb.Organism.filter(ontology_id=o).one()).df()
            organism["organism"] = o
            organismdf.append(organism)

        return pd.concat(organismdf)

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
            cats |= set(LABELS_TOADD.get(label, []))
            groupings, _, lclass = get_ancestry_mapping(cats, parentdf)
            for i, j in groupings.items():
                if len(j) == 0:
                    groupings.pop(i)
            self.class_groupings[label] = groupings
            self.class_topred[label] = lclass
