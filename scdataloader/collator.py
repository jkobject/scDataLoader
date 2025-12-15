from typing import List, Optional

import numpy as np
import pandas as pd
from torch import Tensor, long

from .preprocess import _digitize
from .utils import load_genes


class Collator:
    def __init__(
        self,
        organisms: List[str],
        how: str = "all",
        org_to_id: dict[str, int] = None,
        valid_genes: Optional[List[str]] = None,
        max_len: int = 2000,
        add_zero_genes: int = 0,
        logp1: bool = False,
        norm_to: Optional[float] = None,
        n_bins: int = 0,
        tp_name: Optional[str] = None,
        organism_name: str = "organism_ontology_term_id",
        class_names: List[str] = [],
        genelist: List[str] = [],
        genedf: Optional[pd.DataFrame] = None,
    ):
        """
        Collator for preparing gene expression data batches for the scPRINT model.

        This class handles the organization and preparation of gene expression data from
        different organisms, allowing for various configurations such as maximum gene list
        length, normalization, binning, and gene selection strategies.

        Compatible with scVI's dataloader and other PyTorch data loading pipelines.

        Args:
            organisms (List[str]): List of organism ontology term IDs to include.
                Samples from other organisms will be dropped (may lead to variable batch sizes).
            how (str, optional): Gene selection strategy. Defaults to "all".
                - "most expr": Select the `max_len` most expressed genes. If fewer genes
                  are expressed, randomly sample unexpressed genes to fill.
                - "random expr": Randomly select `max_len` expressed genes. If fewer genes
                  are expressed, randomly sample unexpressed genes to fill.
                - "all": Use all genes without filtering.
                - "some": Use only genes specified in the `genelist` parameter.
            org_to_id (dict[str, int], optional): Mapping from organism names to integer IDs.
                If None, organism names are used directly. Defaults to None.
            valid_genes (List[str], optional): List of gene names to consider from input data.
                Genes not in this list will be dropped. Useful when the model only supports
                specific genes. Defaults to None (use all genes).
            max_len (int, optional): Maximum number of genes to include when using "most expr"
                or "random expr" selection methods. Defaults to 2000.
            add_zero_genes (int, optional): Number of additional unexpressed genes to include
                in the output. Only applies when `how` is "most expr" or "random expr".
                Defaults to 0.
            logp1 (bool, optional): Apply log2(1 + x) transformation to expression values.
                Applied after normalization if both are enabled. Defaults to False.
            norm_to (float, optional): Target sum for count normalization. Expression values
                are scaled so that total counts equal this value. Defaults to None (no normalization).
            n_bins (int, optional): Number of bins for expression value binning. If 0, no
                binning is applied. Binning uses quantile-based discretization. Defaults to 0.
            tp_name (str, optional): Column name in batch data for time point or heat diffusion
                values. If None, time point values default to 0. Defaults to None.
            organism_name (str, optional): Column name in batch data for organism ontology
                term ID. Defaults to "organism_ontology_term_id".
            class_names (List[str], optional): List of additional metadata column names to
                include in the output. Defaults to [].
            genelist (List[str], optional): List of specific genes to use when `how="some"`.
                Required if `how="some"`. Defaults to [].
            genedf (pd.DataFrame, optional): DataFrame containing gene information indexed by
                gene name with an 'organism' column. If None, loaded automatically using
                `load_genes()`. Defaults to None.

        Attributes:
            organism_ids (set): Set of organism IDs being processed.
            start_idx (dict): Mapping from organism ID to starting gene index in the model.
            accepted_genes (dict): Boolean masks for valid genes per organism.
            to_subset (dict): Boolean masks for genelist filtering per organism.

        Raises:
            AssertionError: If `how="some"` but `genelist` is empty.
        """
        self.organisms = organisms
        self.max_len = max_len
        self.n_bins = n_bins
        self.add_zero_genes = add_zero_genes
        self.logp1 = logp1
        self.norm_to = norm_to
        self.how = how
        if self.how == "some":
            assert len(genelist) > 0, "if how is some, genelist must be provided"
        self.organism_name = organism_name
        self.tp_name = tp_name
        self.class_names = class_names
        self.start_idx = {}
        self.accepted_genes = {}
        self.to_subset = {}
        self._setup(genedf, org_to_id, valid_genes, genelist)

    def _setup(self, genedf=None, org_to_id=None, valid_genes=[], genelist=[]):
        """
        Initialize gene mappings and indices for each organism.

        Sets up internal data structures for gene filtering, organism-specific
        gene indices, and gene subsetting based on the provided configuration.

        Args:
            genedf (pd.DataFrame, optional): Gene information DataFrame. If None,
                loaded via `load_genes()`. Defaults to None.
            org_to_id (dict, optional): Organism name to ID mapping. Defaults to None.
            valid_genes (List[str], optional): Genes to accept from input. Defaults to [].
            genelist (List[str], optional): Genes to subset to when `how="some"`. Defaults to [].
        """
        if genedf is None:
            genedf = load_genes(self.organisms)
            self.organism_ids = (
                set([org_to_id[k] for k in self.organisms])
                if org_to_id is not None
                else set(self.organisms)
            )
        self.org_to_id = org_to_id
        self.to_subset = {}
        self.accepted_genes = {}
        self.start_idx = {}

        if valid_genes is not None:
            if len(set(valid_genes) - set(genedf.index)) > 0:
                print("Some valid genes are not in the genedf!!!")
            tot = genedf[genedf.index.isin(valid_genes)]
        else:
            tot = genedf
        for organism in self.organisms:
            org = org_to_id[organism] if org_to_id is not None else organism
            self.start_idx.update({org: np.where(tot.organism == organism)[0][0]})

            ogenedf = genedf[genedf.organism == organism]
            if valid_genes is not None:
                self.accepted_genes.update({org: ogenedf.index.isin(valid_genes)})
            if len(genelist) > 0:
                df = ogenedf[ogenedf.index.isin(valid_genes)]
                self.to_subset.update({org: df.index.isin(genelist)})

    def __call__(self, batch) -> dict[str, Tensor]:
        """
        Collate a minibatch of gene expression data.

        Processes a list of sample dictionaries, applying gene selection, normalization,
        log transformation, and binning as configured. Filters out samples from organisms
        not in the configured organism list.

        Args:
            batch (List[dict]): List of sample dictionaries, each containing:
                - "X" (array): Gene expression values.
                - organism_name (any): Organism identifier (column name set by `organism_name`).
                - tp_name (float, optional): Time point value (column name set by `tp_name`).
                - class_names... (any, optional): Additional class labels.
                - "_storage_idx" (int, optional): Dataset storage index.
                - "is_meta" (int, optional): Metadata flag.
                - "knn_cells" (array, optional): KNN neighbor expression data.
                - "knn_cells_info" (array, optional): KNN neighbor metadata.

        Returns:
            dict[str, Tensor]: Dictionary containing collated tensors:
                - "x" (Tensor): Gene expression matrix of shape (batch_size, n_genes).
                  Values may be raw counts, normalized, log-transformed, or binned
                  depending on configuration.
                - "genes" (Tensor): Gene indices of shape (batch_size, n_genes) as int32.
                  Indices correspond to positions in the model's gene vocabulary.
                - "class" (Tensor): Class labels of shape (batch_size, n_classes) as int32.
                - "tp" (Tensor): Time point values of shape (batch_size,).
                - "depth" (Tensor): Total counts per cell of shape (batch_size,).
                - "is_meta" (Tensor, optional): Metadata flags as int32. Present if input
                  contains "is_meta".
                - "knn_cells" (Tensor, optional): KNN expression data. Present if input
                  contains "knn_cells".
                - "knn_cells_info" (Tensor, optional): KNN metadata. Present if input
                  contains "knn_cells_info".
                - "dataset" (Tensor, optional): Dataset indices as int64. Present if input
                  contains "_storage_idx".

        Note:
            Batch size in output may be smaller than input if some samples are filtered
            out due to organism mismatch.
        """
        # do count selection
        # get the unseen info and don't add any unseen
        # get the I most expressed genes, add randomly some unexpressed genes that are not unseen
        exprs = []
        total_count = []
        other_classes = []
        gene_locs = []
        tp = []
        dataset = []
        nnz_loc = []
        is_meta = []
        knn_cells = []
        knn_cells_info = []
        for elem in batch:
            organism_id = elem[self.organism_name]
            if organism_id not in self.organism_ids:
                continue
            if "_storage_idx" in elem:
                dataset.append(elem["_storage_idx"])
            expr = np.array(elem["X"])
            total_count.append(expr.sum())
            if len(self.accepted_genes) > 0:
                expr = expr[self.accepted_genes[organism_id]]
                if "knn_cells" in elem:
                    elem["knn_cells"] = elem["knn_cells"][
                        :, self.accepted_genes[organism_id]
                    ]
            if self.how == "most expr":
                if "knn_cells" in elem:
                    nnz_loc = np.where(expr + elem["knn_cells"].sum(0) > 0)[0]
                    ma = self.max_len if self.max_len < len(nnz_loc) else len(nnz_loc)
                    loc = np.argsort(expr + elem["knn_cells"].mean(0))[-(ma):][::-1]
                else:
                    nnz_loc = np.where(expr > 0)[0]
                    ma = self.max_len if self.max_len < len(nnz_loc) else len(nnz_loc)
                    loc = np.argsort(expr)[-(ma):][::-1]
                # nnz_loc = [1] * 30_000
                # loc = np.argsort(expr)[-(self.max_len) :][::-1]
            elif self.how == "random expr":
                nnz_loc = np.where(expr > 0)[0]
                loc = (
                    nnz_loc[
                        np.random.choice(
                            len(nnz_loc),
                            self.max_len,
                            replace=False,
                            # p=(expr.max() + (expr[nnz_loc])*19) / expr.max(), # 20 at most times more likely to be selected
                        )
                    ]
                    if self.max_len < len(nnz_loc)
                    else nnz_loc
                )
            elif self.how in ["all", "some"]:
                loc = np.arange(len(expr))
            else:
                raise ValueError("how must be either most expr or random expr")
            if (
                (self.add_zero_genes > 0) or (self.max_len > len(nnz_loc))
            ) and self.how not in [
                "all",
                "some",
            ]:
                ma = self.add_zero_genes + (
                    0 if self.max_len < len(nnz_loc) else self.max_len - len(nnz_loc)
                )
                if "knn_cells" in elem:
                    # we complete with genes expressed in the knn
                    # which is not a zero_loc in this context
                    knn_expr = elem["knn_cells"].sum(0)
                    mask = np.ones(len(knn_expr), dtype=bool)
                    mask[loc] = False
                    available_indices = np.where(mask)[0]
                    available_knn_expr = knn_expr[available_indices]
                    sorted_indices = np.argsort(available_knn_expr)[::-1]
                    selected = min(ma, len(available_indices))
                    zero_loc = available_indices[sorted_indices[:selected]]
                else:
                    zero_loc = np.where(expr == 0)[0]
                    zero_loc = zero_loc[
                        np.random.choice(
                            len(zero_loc),
                            ma,
                            replace=False,
                        )
                    ]
                loc = np.concatenate((loc, zero_loc), axis=None)
            expr = expr[loc]
            if "knn_cells" in elem:
                elem["knn_cells"] = elem["knn_cells"][:, loc]
            if self.how == "some":
                if "knn_cells" in elem:
                    elem["knn_cells"] = elem["knn_cells"][
                        :, self.to_subset[organism_id]
                    ]
                expr = expr[self.to_subset[organism_id]]
                loc = loc[self.to_subset[organism_id]]
            exprs.append(expr)
            if "knn_cells" in elem:
                knn_cells.append(elem["knn_cells"])
            if "knn_cells_info" in elem:
                knn_cells_info.append(elem["knn_cells_info"])
            # then we need to add the start_idx to the loc to give it the correct index
            # according to the model
            gene_locs.append(loc + self.start_idx[organism_id])

            if self.tp_name is not None:
                tp.append(elem[self.tp_name])
            else:
                tp.append(0)
            if "is_meta" in elem:
                is_meta.append(elem["is_meta"])
            other_classes.append([elem[i] for i in self.class_names])
        expr = np.array(exprs)
        tp = np.array(tp)
        gene_locs = np.array(gene_locs)
        total_count = np.array(total_count)
        other_classes = np.array(other_classes)
        dataset = np.array(dataset)
        is_meta = np.array(is_meta)
        knn_cells = np.array(knn_cells)
        knn_cells_info = np.array(knn_cells_info)

        # normalize counts
        if self.norm_to is not None:
            expr = (expr * self.norm_to) / total_count[:, None]
            # TODO: solve issue here
            knn_cells = (knn_cells * self.norm_to) / total_count[:, None]
        if self.logp1:
            expr = np.log2(1 + expr)
            knn_cells = np.log2(1 + knn_cells)

        # do binning of counts
        if self.n_bins > 0:
            binned_rows = []
            bin_edges = []
            for row in expr:
                if row.max() == 0:
                    print(
                        "The input data contains all zero rows. Please make sure "
                        "this is expected. You can use the `filter_cell_by_counts` "
                        "arg to filter out all zero rows."
                    )
                    binned_rows.append(np.zeros_like(row, dtype=np.int64))
                    bin_edges.append(np.array([0] * self.n_bins))
                    continue
                non_zero_ids = row.nonzero()
                non_zero_row = row[non_zero_ids]
                bins = np.quantile(non_zero_row, np.linspace(0, 1, self.n_bins - 1))
                # bins = np.sort(np.unique(bins))
                # NOTE: comment this line for now, since this will make the each category
                # has different relative meaning across datasets
                non_zero_digits = _digitize(non_zero_row, bins)
                assert non_zero_digits.min() >= 1
                assert non_zero_digits.max() <= self.n_bins - 1
                binned_row = np.zeros_like(row, dtype=np.int64)
                binned_row[non_zero_ids] = non_zero_digits
                binned_rows.append(binned_row)
                bin_edges.append(np.concatenate([[0], bins]))
            expr = np.stack(binned_rows)
            # expr = np.digitize(expr, bins=self.bins)

        ret = {
            "x": Tensor(expr),
            "genes": Tensor(gene_locs).int(),
            "class": Tensor(other_classes).int(),
            "tp": Tensor(tp),
            "depth": Tensor(total_count),
        }
        if len(is_meta) > 0:
            ret.update({"is_meta": Tensor(is_meta).int()})
        if len(knn_cells) > 0:
            ret.update({"knn_cells": Tensor(knn_cells)})
        if len(knn_cells_info) > 0:
            ret.update({"knn_cells_info": Tensor(knn_cells_info)})
        if len(dataset) > 0:
            ret.update({"dataset": Tensor(dataset).to(long)})
        return ret
