from scdataloader import utils

import pandas as pd

import lamindb as ln
import lnschema_bionty as lb
import cellxgene_census

from scdataloader.preprocess import (
    LaminPreprocessor,
    additional_postprocess,
    additional_preprocess,
)

import hydra
from config import Config
from hydra.core.config_store import ConfigStore

# convoluted way to work with types
cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(config_path="config", config_name="main")
def main(cfg: Config):

    if cfg.reinit_lamindb:
        ## load some known ontology names

        # first if you use a local instance you will need to populate your ontologies.

        # one could just add everything by keeping the default `None` value for the `ontology` argument, but this will take a long time.

        # Instead, we will load only the ontologies we need. By using all the used/existing cellxgene ontology names.

        census = cellxgene_census.open_soma(census_version="latest")
        val_to_get = [
            "self_reported_ethnicity_ontology_term_id",
            "assay_ontology_term_id",
            "development_stage_ontology_term_id",
            "disease_ontology_term_id",
            "tissue_ontology_term_id",
            "organism_ontology_term_id",
        ]
        df = pd.concat([census["census_data"][organism]
            .obs.read(column_names=val_to_get, value_filter="is_primary_data == True")
            .concat()
            .to_pandas() for organism in cfg.organisms])

        utils.populate_my_ontology(
            organisms=df["organism_ontology_term_id"],
            sex=["PATO:0000384", "PATO:0000383"],
            ethnicities=df["self_reported_ethnicity_ontology_term_id"].unique().tolist(),
            assays=list(
                set(df["assay_ontology_term_id"].unique()).union(set(cfg.additional_assays.keys()))
            ) + ["EFO:0010961"],
            tissues=list(
                set(df["tissue_ontology_term_id"].unique()).union(set(cfg.additional_tissues.keys()))
                )
            ),
            # we load all possible diseases. makes it easier
            # diseases=list(set(df['disease_ontology_term_id'].unique()).union(df2['disease_ontology_term_id'].unique())),
            dev_stages=list(df["development_stage_ontology_term_id"].unique()),
        )

        # in some instances you would have to add missing ontologies in the case where you haven't loaded everything
        #bionty_source_ds_mouse = lb.BiontySource.filter(
        #    entity="DevelopmentalStage", organism="mouse"
        #).one()
        #records = lb.DevelopmentalStage.from_values(
        #    df2["development_stage_ontology_term_id"].unique().tolist(),
        #    field=lb.DevelopmentalStage.ontology_id,
        #    bionty_source=bionty_source_ds_mouse,
        #)
        #ln.save(records)

    cx_dataset = ln.Collection.using("laminlabs/cellxgene").one()
    print(cx_dataset, len(cx_dataset.files.all()))
    warnings.filterwarnings("ignore", category=ResourceWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    do_preprocess = LaminPreprocessor(
        additional_postprocess=additional_postprocess,
        additional_preprocess=additional_preprocess,
    )
    preprocessed_dataset = do_preprocess(
        cx_dataset, start_at=cfg.start_at, description=cfg.description
    )
    # we have processed that many files

    print(len(ln.Artifact.filter(version=cfg.version, description=cfg.description)))
    return preprocessed_dataset


if __name__ == "__main__":
    main()
