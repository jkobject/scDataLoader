from typing import Optional

import numpy as np
from torch import Tensor, long

from .utils import downsample_profile, load_genes


class Collator:
    def __init__(
        self,
        organisms: list[str],
        how: str = "all",
        org_to_id: dict[str, int] = None,
        valid_genes: list[str] = [],
        max_len: int = 2000,
        add_zero_genes: int = 0,
        logp1: bool = False,
        norm_to: Optional[float] = None,
        n_bins: int = 0,
        tp_name: Optional[str] = None,
        organism_name: str = "organism_ontology_term_id",
        class_names: list[str] = [],
        genelist: list[str] = [],
        downsample: Optional[float] = None,  # don't use it for training!
        save_output: Optional[str] = None,
    ):
        """
        This class is responsible for collating data for the scPRINT model. It handles the
        organization and preparation of gene expression data from different organisms,
        allowing for various configurations such as maximum gene list length, normalization,
        and selection method for gene expression.

        This Collator should work with scVI's dataloader as well!

        Args:
            organisms (list): List of organisms to be considered for gene expression data.
                it will drop any other organism it sees (might lead to batches of different sizes!)
            how (flag, optional): Method for selecting gene expression. Defaults to "most expr".
                one of ["most expr", "random expr", "all", "some"]:
                "most expr": selects the max_len most expressed genes,
                if less genes are expressed, will sample random unexpressed genes,
                "random expr": uses a random set of max_len expressed genes.
                if less genes are expressed, will sample random unexpressed genes
                "all": uses all genes
                "some": uses only the genes provided through the genelist param
            org_to_id (dict): Dictionary mapping organisms to their respective IDs.
            valid_genes (list, optional): List of genes from the datasets, to be considered. Defaults to [].
                it will drop any other genes from the input expression data (usefull when your model only works on some genes)
            max_len (int, optional): Total number of genes to use (for random expr and most expr). Defaults to 2000.
            n_bins (int, optional): Number of bins for binning the data. Defaults to 0. meaning, no binning of expression.
            add_zero_genes (int, optional): Number of additional unexpressed genes to add to the input data. Defaults to 0.
            logp1 (bool, optional): If True, logp1 normalization is applied. Defaults to False.
            norm_to (float, optional): Rescaling value of the normalization to be applied. Defaults to None.
            organism_name (str, optional): Name of the organism ontology term id. Defaults to "organism_ontology_term_id".
            tp_name (str, optional): Name of the heat diff. Defaults to None.
            class_names (list, optional): List of other classes to be considered. Defaults to [].
            genelist (list, optional): List of genes to be considered. Defaults to [].
                If [] all genes will be considered
            downsample (float, optional): Downsample the profile to a certain number of cells. Defaults to None.
                This is usually done by the scPRINT model during training but this option allows you to do it directly from the collator
            save_output (str, optional): If not None, saves the output to a file. Defaults to None.
                This is mainly for debugging purposes
        """
        self.organisms = organisms
        self.genedf = load_genes(organisms)
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
        self.save_output = save_output
        self.start_idx = {}
        self.accepted_genes = {}
        self.downsample = downsample
        self.to_subset = {}
        self._setup(org_to_id, valid_genes, genelist)

    def _setup(self, org_to_id=None, valid_genes=[], genelist=[]):
        self.org_to_id = org_to_id
        self.to_subset = {}
        self.accepted_genes = {}
        self.start_idx = {}
        self.organism_ids = (
            set([org_to_id[k] for k in self.organisms])
            if org_to_id is not None
            else set(self.organisms)
        )
        for organism in self.organisms:
            ogenedf = self.genedf[self.genedf.organism == organism]
            if len(valid_genes) > 0:
                tot = self.genedf[self.genedf.index.isin(valid_genes)]
            else:
                tot = self.genedf
            org = org_to_id[organism] if org_to_id is not None else organism
            self.start_idx.update({org: np.where(tot.organism == organism)[0][0]})
            if len(valid_genes) > 0:
                self.accepted_genes.update({org: ogenedf.index.isin(valid_genes)})
            if len(genelist) > 0:
                df = ogenedf[ogenedf.index.isin(valid_genes)]
                self.to_subset.update({org: df.index.isin(genelist)})

    def __call__(self, batch) -> dict[str, Tensor]:
        """
        __call__ applies the collator to a minibatch of data

        Args:
            batch (list[dict[str: array]]): List of dicts of arrays containing gene expression data.
                the first list is for the different samples, the second list is for the different elements with
                elem["X"]: gene expression
                elem["organism_name"]: organism ontology term id
                elem["tp_name"]: heat diff
                elem["class_names.."]: other classes

        Returns:
            list[Tensor]: List of tensors containing the collated data.
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
                nnz_loc = np.where(expr > 0)[0]
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
                loc = nnz_loc[
                    np.random.choice(
                        len(nnz_loc),
                        self.max_len if self.max_len < len(nnz_loc) else len(nnz_loc),
                        replace=False,
                        # p=(expr.max() + (expr[nnz_loc])*19) / expr.max(), # 20 at most times more likely to be selected
                    )
                ]
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
                if "knn_cells" in elem:
                    # we complete with genes expressed in the knn
                    nnz_loc = np.where(elem["knn_cells"].sum(0) > 0)[0]
                    ma = self.max_len if self.max_len < len(nnz_loc) else len(nnz_loc)
                    # which is not a zero_loc in this context
                    zero_loc = np.argsort(elem["knn_cells"].sum(0))[-(ma):][::-1]
                else:
                    zero_loc = np.where(expr == 0)[0]
                    zero_loc = zero_loc[
                        np.random.choice(
                            len(zero_loc),
                            self.add_zero_genes
                            + (
                                0
                                if self.max_len < len(nnz_loc)
                                else self.max_len - len(nnz_loc)
                            ),
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
        # normalize counts
        if self.norm_to is not None:
            expr = (expr * self.norm_to) / total_count[:, None]
        if self.logp1:
            expr = np.log2(1 + expr)

        # do binning of counts
        if self.n_bins:
            pass

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
        if len(dataset) > 0:
            ret.update({"dataset": Tensor(dataset).to(long)})
        if self.downsample is not None:
            ret["x"] = downsample_profile(ret["x"], self.downsample)
        if self.save_output is not None:
            with open(self.save_output, "a") as f:
                np.savetxt(f, ret["x"].numpy())
            with open(self.save_output + "_loc", "a") as f:
                np.savetxt(f, gene_locs)
        return ret


#############
#### WIP ####
#############
class GeneformerCollator(Collator):
    def __init__(self, *args, gene_norm_list: list, **kwargs):
        """
        GeneformerCollator to finish

        Args:
            gene_norm_list (list): the normalization of expression through all datasets, per gene.
        """
        super().__init__(*args, **kwargs)
        self.gene_norm_list = gene_norm_list

    def __call__(self, batch):
        super().__call__(batch)
        # normlization per gene

        # tokenize the empty locations


class scGPTCollator(Collator):
    """
    scGPTCollator to finish
    """

    def __call__(self, batch):
        super().__call__(batch)
        # binning

        # tokenize the empty locations


class scPRINTCollator(Collator):
    def __call__(self, batch):
        super().__call__(batch)
