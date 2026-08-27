from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scdataloader import DataModule
from scdataloader.collator import Collator, build_axis_mapping


def test_gene_embeddings_is_an_explicit_datamodule_cli_parameter():
    assert "gene_embeddings" in inspect.signature(DataModule.__init__).parameters


def test_build_axis_mapping_preserves_model_gene_order():
    accepted, model_positions = build_axis_mapping(
        ["g2", "missing", "g0"], {"g0": 0, "g1": 1, "g2": 2}
    )

    assert accepted.tolist() == [True, False, True]
    assert model_positions.tolist() == [2, 0]


def test_storage_mapping_reorders_expression_and_emits_storage_index():
    gene_frame = pd.DataFrame(
        {"organism": ["org", "org", "org"]}, index=["g0", "g1", "g2"]
    )
    mapping = build_axis_mapping(
        ["g2", "missing", "g0"], {"g0": 0, "g1": 1, "g2": 2}
    )
    collator = Collator(
        organisms=["org"],
        how="all",
        org_to_id={"org": 0},
        valid_genes=["g0", "g1", "g2"],
        max_len=3,
        organism_name="organism_ontology_term_id",
        genedf=gene_frame,
        storage_gene_mappings=[mapping],
    )
    # Collator's existing genedf path does not initialize this derived set.
    collator.organism_ids = {0}

    result = collator(
        [
            {
                "X": np.array([20.0, 99.0, 10.0]),
                "organism_ontology_term_id": 0,
                "_store_idx": 0,
            }
        ]
    )

    assert result["x"].tolist() == [[20.0, 10.0]]
    assert result["genes"].tolist() == [[2, 0]]
    assert result["dataset"].tolist() == [0]
    assert result["depth"].tolist() == [129.0]


def test_genes_dict_uses_embedding_subset_order():
    datamodule = object.__new__(DataModule)
    datamodule.gene_subset = ["g2", "g0", "g3"]
    datamodule.dataset = SimpleNamespace(
        organisms=["org_a", "org_b"],
        genedf=pd.DataFrame(
            {"organism": ["org_a", "org_a", "org_b", "org_b"]},
            index=["g0", "g1", "g2", "g3"],
        ),
    )

    assert datamodule.genes_dict == {
        "org_a": ["g0"],
        "org_b": ["g2", "g3"],
    }
