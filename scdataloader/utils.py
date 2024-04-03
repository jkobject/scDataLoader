import io
import os
import urllib

import bionty as bt
import lamindb as ln
import numpy as np
import pandas as pd
from biomart import BiomartServer
from django.db import IntegrityError
from scipy.sparse import csr_matrix
from scipy.stats import median_abs_deviation
from functools import lru_cache
from collections import Counter

from typing import Union, List, Optional

from anndata import AnnData


def createFoldersFor(filepath: str):
    """
    will recursively create folders if needed until having all the folders required to save the file in this filepath
    """
    prevval = ""
    for val in os.path.expanduser(filepath).split("/")[:-1]:
        prevval += val + "/"
        if not os.path.exists(prevval):
            os.mkdir(prevval)


def _fetchFromServer(
    ensemble_server: str, attributes: list, database: str = "hsapiens_gene_ensembl"
):
    """
    Fetches data from the specified ensemble server.

    Args:
        ensemble_server (str): The URL of the ensemble server to fetch data from.
        attributes (list): The list of attributes to fetch from the server.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the fetched data.
    """
    server = BiomartServer(ensemble_server)
    ensmbl = server.datasets[database]
    print(attributes)
    res = pd.read_csv(
        io.StringIO(
            ensmbl.search({"attributes": attributes}, header=1).content.decode()
        ),
        sep="\t",
    )
    return res


def getBiomartTable(
    ensemble_server: str = "http://jul2023.archive.ensembl.org/biomart",
    useCache: bool = False,
    cache_folder: str = "/tmp/biomart/",
    attributes: List[str] = [],
    bypass_attributes: bool = False,
    database: str = "hsapiens_gene_ensembl",
):
    """generate a genelist dataframe from ensembl's biomart

    Args:
        ensemble_server (str, optional): the biomart server. Defaults to "http://jul2023.archive.ensembl.org/biomart".
        useCache (bool, optional): whether to use the cache or not. Defaults to False.
        cache_folder (str, optional): the cache folder. Defaults to "/tmp/biomart/".

    Raises:
        ValueError: should be a dataframe (when the result from the server is something else)

    Returns:
        pd.DataFrame: the dataframe
    """
    attr = (
        [
            "ensembl_gene_id",
            "hgnc_symbol",
            "gene_biotype",
            "entrezgene_id",
        ]
        if not bypass_attributes
        else []
    )
    assert cache_folder[-1] == "/"

    cache_folder = os.path.expanduser(cache_folder)
    createFoldersFor(cache_folder)
    cachefile = os.path.join(cache_folder, ".biomart.csv")
    if useCache & os.path.isfile(cachefile):
        print("fetching gene names from biomart cache")
        res = pd.read_csv(cachefile)
    else:
        print("downloading gene names from biomart")

        res = _fetchFromServer(ensemble_server, attr + attributes, database=database)
        res.to_csv(cachefile, index=False)

    res.columns = attr + attributes
    if type(res) is not type(pd.DataFrame()):
        raise ValueError("should be a dataframe")
    res = res[~(res["ensembl_gene_id"].isna() & res["hgnc_symbol"].isna())]
    res.loc[res[res.hgnc_symbol.isna()].index, "hgnc_symbol"] = res[
        res.hgnc_symbol.isna()
    ]["ensembl_gene_id"]

    return res


def validate(adata: AnnData, organism: str):
    """
    validate checks if the adata object is valid for lamindb

    Args:
        adata (anndata): the anndata object
        lb (lamindb): the lamindb instance
        organism (str): the organism

    Raises:
        ValueError: if the adata object is not valid
        ValueError: if the anndata contains invalid ethnicity ontology term id according to the lb instance
        ValueError: if the anndata contains invalid organism ontology term id according to the lb instance
        ValueError: if the anndata contains invalid sex ontology term id according to the lb instance
        ValueError: if the anndata contains invalid disease ontology term id according to the lb instance
        ValueError: if the anndata contains invalid cell_type ontology term id according to the lb instance
        ValueError: if the anndata contains invalid development_stage ontology term id according to the lb instance
        ValueError: if the anndata contains invalid tissue ontology term id according to the lb instance
        ValueError: if the anndata contains invalid assay ontology term id according to the lb instance

    Returns:
        bool: True if the adata object is valid
    """
    organism = bt.Organism.filter(ontology_id=organism).one().name

    if adata.var.index.duplicated().any():
        raise ValueError("Duplicate gene names found in adata.var.index")
    if adata.obs.index.duplicated().any():
        raise ValueError("Duplicate cell names found in adata.obs.index")
    for val in [
        "self_reported_ethnicity_ontology_term_id",
        "organism_ontology_term_id",
        "disease_ontology_term_id",
        "cell_type_ontology_term_id",
        "development_stage_ontology_term_id",
        "tissue_ontology_term_id",
        "assay_ontology_term_id",
    ]:
        if val not in adata.obs.columns:
            raise ValueError(
                f"Column '{val}' is missing in the provided anndata object."
            )

    if not bt.Ethnicity.validate(
        adata.obs["self_reported_ethnicity_ontology_term_id"],
        field="ontology_id",
    ).all():
        raise ValueError("Invalid ethnicity ontology term id found")
    if not bt.Organism.validate(
        adata.obs["organism_ontology_term_id"], field="ontology_id"
    ).all():
        raise ValueError("Invalid organism ontology term id found")
    if not bt.Phenotype.validate(
        adata.obs["sex_ontology_term_id"], field="ontology_id"
    ).all():
        raise ValueError("Invalid sex ontology term id found")
    if not bt.Disease.validate(
        adata.obs["disease_ontology_term_id"], field="ontology_id"
    ).all():
        raise ValueError("Invalid disease ontology term id found")
    if not bt.CellType.validate(
        adata.obs["cell_type_ontology_term_id"], field="ontology_id"
    ).all():
        raise ValueError("Invalid cell type ontology term id found")
    if not bt.DevelopmentalStage.validate(
        adata.obs["development_stage_ontology_term_id"],
        field="ontology_id",
    ).all():
        raise ValueError("Invalid dev stage ontology term id found")
    if not bt.Tissue.validate(
        adata.obs["tissue_ontology_term_id"], field="ontology_id"
    ).all():
        raise ValueError("Invalid tissue ontology term id found")
    if not bt.ExperimentalFactor.validate(
        adata.obs["assay_ontology_term_id"], field="ontology_id"
    ).all():
        raise ValueError("Invalid assay ontology term id found")
    if not bt.Gene.validate(
        adata.var.index, field="ensembl_gene_id", organism=organism
    ).all():
        raise ValueError("Invalid gene ensembl id found")
    return True


# setting a cache of 200 elements
# @lru_cache(maxsize=200)
def get_all_ancestors(val: str, df: pd.DataFrame):
    if val not in df.index:
        return set()
    parents = df.loc[val].parents__ontology_id
    if parents is None or len(parents) == 0:
        return set()
    else:
        return set.union(set(parents), *[get_all_ancestors(val, df) for val in parents])


def get_ancestry_mapping(all_elem: list, onto_df: pd.DataFrame):
    """
    This function generates a mapping of all elements to their ancestors in the ontology dataframe.

    Args:
        all_elem (list): A list of all elements.
        onto_df (DataFrame): The ontology dataframe.

    Returns:
        dict: A dictionary mapping each element to its ancestors.
    """
    ancestors = {}
    full_ancestors = set()
    for val in all_elem:
        ancestors[val] = get_all_ancestors(val, onto_df) - set([val])

    for val in ancestors.values():
        full_ancestors |= set(val)
    # removing ancestors that are not in our datasets
    full_ancestors = full_ancestors & set(ancestors.keys())
    leafs = set(all_elem) - full_ancestors
    full_ancestors = full_ancestors - leafs

    groupings = {}
    for val in full_ancestors:
        groupings[val] = set()
    for leaf in leafs:
        for ancestor in ancestors[leaf]:
            if ancestor in full_ancestors:
                groupings[ancestor].add(leaf)

    return groupings, full_ancestors, leafs


def load_dataset_local(
    remote_dataset: ln.Collection,
    download_folder: str,
    name: str,
    description: str,
    use_cache: bool = True,
    only: Optional[List[int]] = None,
):
    """
    This function loads a remote lamindb dataset to local.

    Args:
        lb (lamindb): The lamindb instance.
        remote_dataset (lamindb.Dataset): The remote Dataset.
        download_folder (str): The path to the download folder.
        name (str): The name of the dataset.
        description (str): The description of the dataset.
        use_cache (bool, optional): Whether to use cache. Defaults to True.
        only (list, optional): A list of indices to specify which files to download. Defaults to None.

    Returns:
        lamindb.Dataset: The local dataset.
    """
    saved_files = []
    default_storage = ln.Storage.filter(root=ln.settings.storage.as_posix()).one()
    files = (
        remote_dataset.artifacts.all()
        if not only
        else remote_dataset.artifacts.all()[only[0] : only[1]]
    )
    for file in files:
        organism = list(set([i.ontology_id for i in file.organism.all()]))
        if len(organism) > 1:
            print(organism)
            print("Multiple organisms detected")
            continue
        if len(organism) == 0:
            print("No organism detected")
            continue
        organism = bt.Organism.filter(ontology_id=organism[0]).one().name
        # bt.settings.organism = organism
        path = file.path
        try:
            file.save()
        except IntegrityError:
            print(f"File {file.key} already exists in storage")
        # if location already has a file, don't save again
        if use_cache and os.path.exists(os.path.expanduser(download_folder + file.key)):
            print(f"File {file.key} already exists in storage")
        else:
            path.download_to(download_folder + file.key)
        file.storage = default_storage
        try:
            file.save()
        except IntegrityError:
            print(f"File {file.key} already exists in storage")
        saved_files.append(file)
    dataset = ln.Collection(saved_files, name=name, description=description)
    dataset.save()
    return dataset


def load_genes(organisms: Union[str, list] = "NCBITaxon:9606"):  # "NCBITaxon:10090",
    organismdf = []
    if type(organisms) == str:
        organisms = [organisms]
    for organism in organisms:
        genesdf = bt.Gene.filter(
            organism_id=bt.Organism.filter(ontology_id=organism).first().id
        ).df()
        genesdf = genesdf[~genesdf["public_source_id"].isna()]
        genesdf = genesdf.drop_duplicates(subset="ensembl_gene_id")
        genesdf = genesdf.set_index("ensembl_gene_id").sort_index()
        # mitochondrial genes
        genesdf["mt"] = genesdf.symbol.astype(str).str.startswith("MT-")
        # ribosomal genes
        genesdf["ribo"] = genesdf.symbol.astype(str).str.startswith(("RPS", "RPL"))
        # hemoglobin genes.
        genesdf["hb"] = genesdf.symbol.astype(str).str.contains(("^HB[^(P)]"))
        genesdf["organism"] = organism
        organismdf.append(genesdf)
    return pd.concat(organismdf)


def populate_my_ontology(
    organisms: List[str] = ["NCBITaxon:10090", "NCBITaxon:9606"],
    sex: List[str] = ["PATO:0000384", "PATO:0000383"],
    celltypes: List[str] = [],
    ethnicities: List[str] = [],
    assays: List[str] = [],
    tissues: List[str] = [],
    diseases: List[str] = [],
    dev_stages: List[str] = [],
):
    """
    creates a local version of the lamin ontologies and add the required missing values in base ontologies

    run this function just one for each new lamin storage

    erase everything with bt.$ontology.filter().delete()

    add whatever value you need afterward like it is done here with:

    `bt.$ontology(name="ddd", ontology_id="ddddd").save()`

    `df["assay_ontology_term_id"].unique()`

    Args:
        lb (lamindb): lamindb instance.
        organisms (list, optional): List of organisms. Defaults to ["NCBITaxon:10090", "NCBITaxon:9606"].
        sex (list, optional): List of sexes. Defaults to ["PATO:0000384", "PATO:0000383"].
        celltypes (list, optional): List of cell types. Defaults to [].
        ethnicities (list, optional): List of ethnicities. Defaults to [].
        assays (list, optional): List of assays. Defaults to [].
        tissues (list, optional): List of tissues. Defaults to [].
        diseases (list, optional): List of diseases. Defaults to [].
        dev_stages (list, optional): List of developmental stages. Defaults to [].
    """

    names = bt.CellType.public().df().index if not celltypes else celltypes
    records = bt.CellType.from_values(names, field="ontology_id")
    ln.save(records, parents=bool(celltypes))
    bt.CellType(name="unknown", ontology_id="unknown").save()
    # Organism
    names = bt.Organism.public().df().index if not organisms else organisms
    records = [
        i[0] if type(i) is list else i
        for i in [bt.Organism.from_public(ontology_id=i) for i in names]
    ]
    ln.save(records, parents=bool(organisms))
    bt.Organism(name="unknown", ontology_id="unknown").save()
    # Phenotype
    names = bt.Phenotype.public().df().index if not sex else sex
    records = [
        bt.Phenotype.from_public(
            ontology_id=i,
            public_source=bt.PublicSource.filter(
                entity="Phenotype", source="pato"
            ).one(),
        )
        for i in names
    ]
    ln.save(records, parents=bool(sex))
    bt.Phenotype(name="unknown", ontology_id="unknown").save()
    # ethnicity
    names = bt.Ethnicity.public().df().index if not ethnicities else ethnicities
    records = bt.Ethnicity.from_values(names, field="ontology_id")
    ln.save(records, parents=bool(ethnicities))
    bt.Ethnicity(
        name="unknown", ontology_id="unknown"
    ).save()  # multi ethnic will have to get renamed
    # ExperimentalFactor
    names = bt.ExperimentalFactor.public().df().index if not assays else assays
    records = bt.ExperimentalFactor.from_values(names, field="ontology_id")
    ln.save(records, parents=bool(assays))
    bt.ExperimentalFactor(name="unknown", ontology_id="unknown").save()
    # lookup = bt.ExperimentalFactor.lookup()
    # lookup.smart_seq_v4.parents.add(lookup.smart_like)
    # Tissue
    names = bt.Tissue.public().df().index if not tissues else tissues
    records = bt.Tissue.from_values(names, field="ontology_id")
    ln.save(records, parents=bool(tissues))
    bt.Tissue(name="unknown", ontology_id="unknown").save()
    # DevelopmentalStage
    names = (
        bt.DevelopmentalStage.public().df().index if not dev_stages else dev_stages
    )
    records = bt.DevelopmentalStage.from_values(names, field="ontology_id")
    ln.save(records, parents=bool(dev_stages))
    bt.DevelopmentalStage(name="unknown", ontology_id="unknown").save()

    names = bt.DevelopmentalStage.public(organism="mouse").df().name
    bionty_source = bt.PublicSource.filter(
        entity="DevelopmentalStage", organism="mouse"
    ).one()
    records = [
        bt.DevelopmentalStage.from_public(name=i, public_source=bionty_source)
        for i in names.tolist()
    ]
    records[-4] = records[-4][0]
    ln.save(records)
    # Disease
    names = bt.Disease.public().df().index if not diseases else diseases
    records = bt.Disease.from_values(names, field="ontology_id")
    ln.save(records, parents=bool(diseases))
    bt.Disease(name="normal", ontology_id="PATO:0000461").save()
    bt.Disease(name="unknown", ontology_id="unknown").save()
    # genes
    for organism in ["NCBITaxon:10090", "NCBITaxon:9606"]:
        # convert onto to name
        organism = bt.Organism.filter(ontology_id=organism).one().name
        names = bt.Gene.public(organism=organism).df()["ensembl_gene_id"]
        records = bt.Gene.from_values(
            names,
            field="ensembl_gene_id",
            organism=organism,
        )
        ln.save(records)


def is_outlier(adata: AnnData, metric: str, nmads: int):
    """
    is_outlier detects outliers in adata.obs[metric]

    Args:
        adata (annData): the anndata object
        metric (str): the metric column to use
        nmads (int): the number of median absolute deviations to use as a threshold

    Returns:
        pd.Series: a boolean series indicating whether a cell is an outlier or not
    """
    M = adata.obs[metric]
    outlier = (M < np.median(M) - nmads * median_abs_deviation(M)) | (
        np.median(M) + nmads * median_abs_deviation(M) < M
    )
    return outlier


def length_normalize(adata: AnnData, gene_lengths: list):
    """
    length_normalize normalizes the counts by the gene length

    Args:
        adata (anndata): the anndata object
        gene_lengths (list): the gene lengths

    Returns:
        anndata: the anndata object
    """
    adata.X = csr_matrix((adata.X.T / gene_lengths).T)
    return adata


def pd_load_cached(url: str, loc: str = "/tmp/", cache: bool = True, **kwargs):
    """
    pd_load_cached downloads a file from a url and loads it as a pandas dataframe

    Args:
        url (str): the url to download the file from
        loc (str, optional): the location to save the file to. Defaults to "/tmp/".
        cache (bool, optional): whether to use the cached file or not. Defaults to True.

    Returns:
        pd.DataFrame: the dataframe
    """
    # Check if the file exists, if not, download it
    loc += url.split("/")[-1]
    if not os.path.isfile(loc) or not cache:
        urllib.request.urlretrieve(url, loc)
    # Load the data from the file
    return pd.read_csv(loc, **kwargs)


def translate(
    val: Union[str, list, set, Counter, dict], t: str = "cell_type_ontology_term_id"
):
    """
    translate translates the ontology term id to the name

    Args:
        val (str, dict, set, list, dict): the object to translate
        t (flat, optional): the type of ontology terms.
            one of cell_type_ontology_term_id, assay_ontology_term_id, tissue_ontology_term_id.
            Defaults to "cell_type_ontology_term_id".

    Returns:
        dict: the mapping for the translation
    """
    if t == "cell_type_ontology_term_id":
        obj = bt.CellType.public(organism="all")
    elif t == "assay_ontology_term_id":
        obj = bt.ExperimentalFactor.public()
    elif t == "tissue_ontology_term_id":
        obj = bt.Tissue.public()
    else:
        return None
    if type(val) is str:
        return {val: obj.search(val, field=obj.ontology_id).name.iloc[0]}
    elif type(val) is list or type(val) is set:
        return {i: obj.search(i, field=obj.ontology_id).name.iloc[0] for i in set(val)}
    elif type(val) is dict or type(val) is Counter:
        return {
            obj.search(k, field=obj.ontology_id).name.iloc[0]: v for k, v in val.items()
        }
