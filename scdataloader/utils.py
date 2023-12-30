import os
import lamindb as ln
import bionty as bt
from scipy.sparse import csr_matrix
import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import median_abs_deviation
from django.db import IntegrityError


def getBiomartTable(
    ensemble_server="http://jul2023.archive.ensembl.org/biomart",
    useCache=False,
    cache_folder="/tmp/biomart/",
    attributes=[],
    bypass_attributes=False,
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

        res = _fetchFromServer(ensemble_server, attr + attributes)
        res.to_csv(cachefile, index=False)

    res.columns = attr + attributes
    if type(res) is not type(pd.DataFrame()):
        raise ValueError("should be a dataframe")
    res = res[~(res["ensembl_gene_id"].isna() & res["hgnc_symbol"].isna())]
    res.loc[res[res.hgnc_symbol.isna()].index, "hgnc_symbol"] = res[
        res.hgnc_symbol.isna()
    ]["ensembl_gene_id"]

    return res


def validate(adata, lb, organism):
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
    organism = lb.Organism.filter(ontology_id=organism).one().name
    lb.settings.organism = organism

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
    bionty_source = lb.BiontySource.filter(
        entity="DevelopmentalStage", organism=organism
    ).one()

    if not lb.Ethnicity.validate(
        adata.obs["self_reported_ethnicity_ontology_term_id"], field="ontology_id"
    ).all():
        raise ValueError("Invalid ethnicity ontology term id found")
    if not lb.Organism.validate(
        adata.obs["organism_ontology_term_id"], field="ontology_id"
    ).all():
        raise ValueError("Invalid organism ontology term id found")
    if not lb.Phenotype.validate(
        adata.obs["sex_ontology_term_id"], field="ontology_id"
    ).all():
        raise ValueError("Invalid sex ontology term id found")
    if not lb.Disease.validate(
        adata.obs["disease_ontology_term_id"], field="ontology_id"
    ).all():
        raise ValueError("Invalid disease ontology term id found")
    if not lb.CellType.validate(
        adata.obs["cell_type_ontology_term_id"], field="ontology_id"
    ).all():
        raise ValueError("Invalid cell type ontology term id found")
    if (
        not lb.DevelopmentalStage.filter(bionty_source=bionty_source)
        .validate(adata.obs["development_stage_ontology_term_id"], field="ontology_id")
        .all()
    ):
        raise ValueError("Invalid dev stage ontology term id found")
    if not lb.Tissue.validate(
        adata.obs["tissue_ontology_term_id"], field="ontology_id"
    ).all():
        raise ValueError("Invalid tissue ontology term id found")
    if not lb.ExperimentalFactor.validate(
        adata.obs["assay_ontology_term_id"], field="ontology_id"
    ).all():
        raise ValueError("Invalid assay ontology term id found")
    if (
        not lb.Gene.filter(organism=lb.settings.organism)
        .validate(adata.var.index, field="ensembl_gene_id")
        .all()
    ):
        raise ValueError("Invalid gene ensembl id found")
    return True


def get_all_ancestors(val, df):
    if val not in df.index:
        return set()
    parents = df.loc[val].parents__ontology_id
    if parents is None or len(parents) == 0:
        return set()
    else:
        return set.union(set(parents), *[get_all_ancestors(val, df) for val in parents])


def get_ancestry_mapping(all_elem, onto_df):
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
    lb, remote_dataset, download_folder, name, description, use_cache=True, only=None
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
        organism = lb.Organism.filter(ontology_id=organism[0]).one().name
        # lb.settings.organism = organism
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
    dataset = ln.Dataset(saved_files, name=name, description=description)
    dataset.save()
    return dataset


def populate_my_ontology(
    lb,
    organisms=["NCBITaxon:10090", "NCBITaxon:9606"],
    sex=["PATO:0000384", "PATO:0000383"],
    celltypes=[],
    ethnicities=[],
    assays=[],
    tissues=[],
    diseases=[],
    dev_stages=[],
):
    """
    creates a local version of the lamin ontologies and add the required missing values in base ontologies

    run this function just one for each new lamin storage

    erase everything with lb.$ontology.filter().delete()

    add whatever value you need afterward like it is done here with:

    `lb.$ontology(name="ddd", ontology_id="ddddd").save()`

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

    names = bt.CellType().df().index if not celltypes else celltypes
    records = lb.CellType.from_values(names, field=lb.CellType.ontology_id)
    ln.save(records)
    lb.CellType(name="unknown", ontology_id="unknown").save()
    # Organism
    # names = bt.Organism().df().index if not organisms else organisms
    # records = lb.Organism.from_values(names, field=lb.Organism.ontology_id)
    # ln.save(records)
    # lb.Organism(name="unknown", ontology_id="unknown").save()
    # Phenotype
    name = bt.Phenotype().df().index if not sex else sex
    records = lb.Phenotype.from_values(
        name,
        field=lb.Phenotype.ontology_id,
        bionty_source=lb.BiontySource.filter(entity="Phenotype", source="pato").one(),
    )
    ln.save(records)
    lb.Phenotype(name="unknown", ontology_id="unknown").save()
    # ethnicity
    names = bt.Ethnicity().df().index if not ethnicities else ethnicities
    records = lb.Ethnicity.from_values(names, field=lb.Ethnicity.ontology_id)
    ln.save(records)
    lb.Ethnicity(
        name="unknown", ontology_id="unknown"
    ).save()  # multi ethnic will have to get renamed
    # ExperimentalFactor
    names = bt.ExperimentalFactor().df().index if not assays else assays
    records = lb.ExperimentalFactor.from_values(
        names, field=lb.ExperimentalFactor.ontology_id
    )
    ln.save(records)
    lb.ExperimentalFactor(name="unknown", ontology_id="unknown").save()
    # lookup = lb.ExperimentalFactor.lookup()
    # lookup.smart_seq_v4.parents.add(lookup.smart_like)
    # Tissue
    names = bt.Tissue().df().index if not tissues else tissues
    records = lb.Tissue.from_values(names, field=lb.Tissue.ontology_id)
    ln.save(records)
    lb.Tissue(name="unknown", ontology_id="unknown").save()
    # DevelopmentalStage
    names = bt.DevelopmentalStage().df().index if not dev_stages else dev_stages
    records = lb.DevelopmentalStage.from_values(
        names, field=lb.DevelopmentalStage.ontology_id
    )
    ln.save(records)
    lb.DevelopmentalStage(name="unknown", ontology_id="unknown").save()
    # Disease
    names = bt.Disease().df().index if not diseases else diseases
    records = lb.Disease.from_values(names, field=lb.Disease.ontology_id)
    ln.save(records)
    lb.Disease(name="normal", ontology_id="PATO:0000461").save()
    lb.Disease(name="unknown", ontology_id="unknown").save()
    # genes
    for organism in organisms:
        # convert onto to name
        organism = lb.Organism.filter(ontology_id=organism).one().name
        names = bt.Gene(organism=organism).df()["ensembl_gene_id"]
        records = lb.Gene.from_values(
            names,
            field="ensembl_gene_id",
            bionty_source=lb.BiontySource.filter(
                entity="Gene", organism=organism
            ).first(),
        )
        ln.save(records)


def is_outlier(adata, metric: str, nmads: int):
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


def length_normalize(adata, gene_lengths):
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


def pd_load_cached(url, loc="/tmp/", cache=True, **kwargs):
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
