import argparse
from typing import Optional, Union

import lamindb as ln

from scdataloader.preprocess import (
    LaminPreprocessor,
    additional_postprocess,
    additional_preprocess,
)


# scdataloader --instance="laminlabs/cellxgene" --name="cellxgene-census" --version="2023-12-15" \
# --description="scPRINT-V2 datasets" --new_name="scprint v2" --n_hvg_for_postp=4000 --cache=False \
# --filter_gene_by_counts=0 --filter_cell_by_counts=300 --min_valid_genes_id=500 \
# --min_nnz_genes=120 --min_dataset_size=100 --maxdropamount=90 \
# --organisms=["NCBITaxon:9606","NCBITaxon:9544","NCBITaxon:9483","NCBITaxon:10090"] \
# --start_at=0
def main():
    """
    Main function to either preprocess datasets in a lamindb collection or populate ontologies.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess datasets or populate ontologies."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Preprocess command
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess datasets")
    preprocess_parser.add_argument(
        "--name", type=str, required=True, help="Name of the input dataset"
    )
    preprocess_parser.add_argument(
        "--new_name",
        type=str,
        default="preprocessed dataset",
        help="Name of the preprocessed dataset.",
    )
    preprocess_parser.add_argument(
        "--description",
        type=str,
        default="preprocessed by scDataLoader",
        help="Description of the preprocessed dataset.",
    )
    preprocess_parser.add_argument(
        "--start_at", type=int, default=0, help="Position to start preprocessing at."
    )
    preprocess_parser.add_argument(
        "--new_version",
        type=str,
        default="2",
        help="Version of the output dataset and files.",
    )
    preprocess_parser.add_argument(
        "--instance",
        type=str,
        default=None,
        help="Instance storing the input dataset, if not local",
    )
    preprocess_parser.add_argument(
        "--version", type=str, default=None, help="Version of the input dataset."
    )
    preprocess_parser.add_argument(
        "--filter_gene_by_counts",
        type=int,
        default=0,
        help="Determines whether to filter genes by counts.",
    )
    preprocess_parser.add_argument(
        "--filter_cell_by_counts",
        type=int,
        default=0,
        help="Determines whether to filter cells by counts.",
    )
    preprocess_parser.add_argument(
        "--normalize_sum",
        type=float,
        default=1e4,
        help="Determines whether to normalize the total counts of each cell to a specific value.",
    )
    preprocess_parser.add_argument(
        "--n_hvg_for_postp",
        type=int,
        default=0,
        help="Determines whether to subset highly variable genes.",
    )
    preprocess_parser.add_argument(
        "--hvg_flavor",
        type=str,
        default="seurat_v3",
        help="Specifies the flavor of highly variable genes selection.",
    )
    preprocess_parser.add_argument(
        "--binning",
        type=Optional[int],
        default=None,
        help="Determines whether to bin the data into discrete values of number of bins provided.",
    )
    preprocess_parser.add_argument(
        "--result_binned_key",
        type=str,
        default="X_binned",
        help="Specifies the key of AnnData to store the binned data.",
    )
    preprocess_parser.add_argument(
        "--length_normalize",
        type=bool,
        default=False,
        help="Determines whether to normalize the length.",
    )
    preprocess_parser.add_argument(
        "--force_preprocess",
        type=bool,
        default=False,
        help="Determines whether to force preprocessing.",
    )
    preprocess_parser.add_argument(
        "--min_dataset_size",
        type=int,
        default=100,
        help="Specifies the minimum dataset size.",
    )
    preprocess_parser.add_argument(
        "--min_valid_genes_id",
        type=int,
        default=10_000,
        help="Specifies the minimum valid genes id.",
    )
    preprocess_parser.add_argument(
        "--min_nnz_genes",
        type=int,
        default=200,
        help="Specifies the minimum non-zero genes.",
    )
    preprocess_parser.add_argument(
        "--maxdropamount",
        type=int,
        default=50,
        help="Specifies the maximum drop amount.",
    )
    preprocess_parser.add_argument(
        "--madoutlier", type=int, default=5, help="Specifies the MAD outlier."
    )
    preprocess_parser.add_argument(
        "--pct_mt_outlier",
        type=int,
        default=8,
        help="Specifies the percentage of MT outlier.",
    )
    preprocess_parser.add_argument(
        "--batch_keys",
        type=list[str],
        default=[
            "assay_ontology_term_id",
            "self_reported_ethnicity_ontology_term_id",
            "sex_ontology_term_id",
            "donor_id",
            "suspension_type",
        ],
        help="Specifies the batch keys.",
    )
    preprocess_parser.add_argument(
        "--skip_validate",
        type=bool,
        default=False,
        help="Determines whether to skip validation.",
    )
    preprocess_parser.add_argument(
        "--do_postp",
        type=bool,
        default=True,
        help="Determines whether to do postprocessing.",
    )
    preprocess_parser.add_argument(
        "--cache",
        type=bool,
        default=False,
        help="Determines whether to cache the dataset.",
    )
    preprocess_parser.add_argument(
        "--organisms",
        type=list,
        default=[
            "NCBITaxon:9606",
            "NCBITaxon:10090",
        ],
        help="Determines the organisms to keep.",
    )
    preprocess_parser.add_argument(
        "--force_preloaded",
        type=bool,
        default=False,
        help="Determines whether the dataset is preloaded.",
    )
    # Populate command
    populate_parser = subparsers.add_parser("populate", help="Populate ontologies")
    populate_parser.add_argument(
        "what",
        nargs="?",
        default="all",
        choices=[
            "all",
            "organisms",
            "celltypes",
            "diseases",
            "tissues",
            "assays",
            "ethnicities",
            "sex",
            "dev_stages",
        ],
        help="What ontologies to populate",
    )
    args = parser.parse_args()

    if args.command == "populate":
        from scdataloader.utils import populate_my_ontology

        if args.what != "all":
            raise ValueError("Only 'all' is supported for now")
        else:
            populate_my_ontology()
        return

    # Load the collection
    # if not args.preprocess:
    #    print("Only preprocess is available for now")
    #    return
    if args.instance is not None:
        collection = (
            ln.Collection.using(instance=args.instance)
            .filter(name=args.name, version=args.version)
            .first()
        )
    else:
        collection = ln.Collection.filter(name=args.name, version=args.version).first()

    print(
        "using the dataset ", collection, " of size ", len(collection.artifacts.all())
    )
    # Initialize the preprocessor
    preprocessor = LaminPreprocessor(
        filter_gene_by_counts=args.filter_gene_by_counts,
        filter_cell_by_counts=args.filter_cell_by_counts,
        normalize_sum=args.normalize_sum,
        n_hvg_for_postp=args.n_hvg_for_postp,
        hvg_flavor=args.hvg_flavor,
        cache=args.cache,
        binning=args.binning,
        result_binned_key=args.result_binned_key,
        length_normalize=args.length_normalize,
        force_preprocess=args.force_preprocess,
        min_dataset_size=args.min_dataset_size,
        min_valid_genes_id=args.min_valid_genes_id,
        min_nnz_genes=args.min_nnz_genes,
        maxdropamount=args.maxdropamount,
        madoutlier=args.madoutlier,
        pct_mt_outlier=args.pct_mt_outlier,
        batch_keys=args.batch_keys,
        skip_validate=args.skip_validate,
        do_postp=args.do_postp,
        additional_preprocess=additional_preprocess,
        additional_postprocess=additional_postprocess,
        keep_files=False,
        force_preloaded=args.force_preloaded,
    )

    # Preprocess the dataset
    preprocessor(
        collection,
        name=args.new_name,
        description=args.description,
        start_at=args.start_at,
        version=args.new_version,
    )

    print(
        f"Preprocessed dataset saved with version {args.version} and name {args.new_name}."
    )


if __name__ == "__main__":
    main()
