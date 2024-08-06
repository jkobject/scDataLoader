import argparse
from scdataloader.preprocess import (
    LaminPreprocessor,
    additional_preprocess,
    additional_postprocess,
)
import lamindb as ln
from typing import Optional, Union


# scdataloader --instance="laminlabs/cellxgene" --name="cellxgene-census" --version="2023-12-15" --description="preprocessed for scprint" --new_name="scprint main" --start_at=39
def main():
    """
    main function to preprocess datasets in a given lamindb collection.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess datasets in a given lamindb collection."
    )
    parser.add_argument(
        "--name", type=str, required=True, help="Name of the input dataset"
    )
    parser.add_argument(
        "--new_name",
        type=str,
        default="preprocessed dataset",
        help="Name of the preprocessed dataset.",
    )
    parser.add_argument(
        "--description",
        type=str,
        default="preprocessed by scDataLoader",
        help="Description of the preprocessed dataset.",
    )
    parser.add_argument(
        "--start_at", type=int, default=0, help="Position to start preprocessing at."
    )
    parser.add_argument(
        "--new_version",
        type=str,
        default="2",
        help="Version of the output dataset and files.",
    )
    parser.add_argument(
        "--instance",
        type=str,
        default=None,
        help="Instance storing the input dataset, if not local",
    )
    parser.add_argument(
        "--version", type=str, default=None, help="Version of the input dataset."
    )
    parser.add_argument(
        "--filter_gene_by_counts",
        type=Union[int, bool],
        default=False,
        help="Determines whether to filter genes by counts.",
    )
    parser.add_argument(
        "--filter_cell_by_counts",
        type=Union[int, bool],
        default=False,
        help="Determines whether to filter cells by counts.",
    )
    parser.add_argument(
        "--normalize_sum",
        type=float,
        default=1e4,
        help="Determines whether to normalize the total counts of each cell to a specific value.",
    )
    parser.add_argument(
        "--subset_hvg",
        type=int,
        default=0,
        help="Determines whether to subset highly variable genes.",
    )
    parser.add_argument(
        "--hvg_flavor",
        type=str,
        default="seurat_v3",
        help="Specifies the flavor of highly variable genes selection.",
    )
    parser.add_argument(
        "--binning",
        type=Optional[int],
        default=None,
        help="Determines whether to bin the data into discrete values of number of bins provided.",
    )
    parser.add_argument(
        "--result_binned_key",
        type=str,
        default="X_binned",
        help="Specifies the key of AnnData to store the binned data.",
    )
    parser.add_argument(
        "--length_normalize",
        type=bool,
        default=False,
        help="Determines whether to normalize the length.",
    )
    parser.add_argument(
        "--force_preprocess",
        type=bool,
        default=False,
        help="Determines whether to force preprocessing.",
    )
    parser.add_argument(
        "--min_dataset_size",
        type=int,
        default=100,
        help="Specifies the minimum dataset size.",
    )
    parser.add_argument(
        "--min_valid_genes_id",
        type=int,
        default=10_000,
        help="Specifies the minimum valid genes id.",
    )
    parser.add_argument(
        "--min_nnz_genes",
        type=int,
        default=400,
        help="Specifies the minimum non-zero genes.",
    )
    parser.add_argument(
        "--maxdropamount",
        type=int,
        default=50,
        help="Specifies the maximum drop amount.",
    )
    parser.add_argument(
        "--madoutlier", type=int, default=5, help="Specifies the MAD outlier."
    )
    parser.add_argument(
        "--pct_mt_outlier",
        type=int,
        default=8,
        help="Specifies the percentage of MT outlier.",
    )
    parser.add_argument(
        "--batch_key", type=Optional[str], default=None, help="Specifies the batch key."
    )
    parser.add_argument(
        "--skip_validate",
        type=bool,
        default=False,
        help="Determines whether to skip validation.",
    )
    parser.add_argument(
        "--do_postp",
        type=bool,
        default=False,
        help="Determines whether to do postprocessing.",
    )
    args = parser.parse_args()

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
        subset_hvg=args.subset_hvg,
        hvg_flavor=args.hvg_flavor,
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
        batch_key=args.batch_key,
        skip_validate=args.skip_validate,
        do_postp=args.do_postp,
        additional_preprocess=additional_preprocess,
        additional_postprocess=additional_postprocess,
        keep_files=False,
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
