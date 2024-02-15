import numpy as np
from .utils import load_genes
from torch import Tensor

# class SimpleCollator:


class Collator:
    def __init__(
        self,
        organisms: list,
        org_to_id: dict = None,
        valid_genes: list = [],
        max_len=2000,
        n_bins=0,
        add_zero_genes=200,
        logp1=False,
        norm_to=None,
        how="all",
        tp_name=None,
        organism_name="organism_ontology_term_id",
        class_names=[],
    ):
        """
        This class is responsible for collating data for the scPRINT model. It handles the
        organization and preparation of gene expression data from different organisms,
        allowing for various configurations such as maximum gene list length, normalization,
        and selection method for gene expression.

        Args:
            organisms (list): List of organisms to be considered for gene expression data.
            org_to_id (dict): Dictionary mapping organisms to their respective IDs.
            labels (list, optional): List of labels for the data. Defaults to [].
            valid_genes (list, optional): List of genes from the datasets, to be considered. Defaults to [].
            max_len (int, optional): Maximum length of the gene list. Defaults to 2000.
            n_bins (int, optional): Number of bins for binning the data. Defaults to 0.
            add_zero_genes (int, optional): Number of zero genes to add. Defaults to 200.
            logp1 (bool, optional): If True, logp1 normalization is applied. Defaults to False.
            norm_to (str, optional): Normalization method to be applied. Defaults to None.
            how (str, optional): Method for selecting gene expression. Defaults to "most expr".
        """
        self.organisms = organisms
        self.valid_genes = valid_genes
        self.max_len = max_len
        self.n_bins = n_bins
        self.add_zero_genes = add_zero_genes
        self.logp1 = logp1
        self.norm_to = norm_to
        self.org_to_id = org_to_id
        self.how = how
        self.organism_ids = (
            set([org_to_id[k] for k in organisms])
            if org_to_id is not None
            else set(organisms)
        )
        self.organism_name = organism_name
        self.tp_name = tp_name
        self.class_names = class_names

        self.start_idx = {}
        self.accepted_genes = {}
        self.genedf = load_genes(organisms)
        for organism in set(self.genedf.organism):
            ogenedf = self.genedf[self.genedf.organism == organism]
            org = org_to_id[organism] if org_to_id is not None else organism
            self.start_idx.update(
                {org: np.where(self.genedf.organism == organism)[0][0]}
            )
            if len(valid_genes) > 0:
                self.accepted_genes.update({org: ogenedf.index.isin(valid_genes)})

    def __call__(self, batch):
        """
        __call__ is a special method in Python that is called when an instance of the class is called.

        Args:
            batch (list[dict[str: array]]): List of dicts of arrays containing gene expression data.
                the first list is for the different samples, the second list is for the different elements with
                elem["x"]: gene expression
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
        for elem in batch:
            organism_id = elem[self.organism_name]
            if organism_id not in self.organism_ids:
                continue
            expr = np.array(elem["x"])
            total_count.append(expr.sum())
            if len(self.accepted_genes) > 0:
                expr = expr[self.accepted_genes[organism_id]]
            if self.how == "most expr":
                loc = np.argsort(expr)[-(self.max_len) :][::-1]
            elif self.how == "random expr":
                nnz_loc = np.where(expr > 0)[0]
                loc = nnz_loc[
                    np.random.choice(len(nnz_loc), self.max_len, replace=False)
                ]
            elif self.how == "all":
                loc = np.arange(len(expr))
            else:
                raise ValueError("how must be either most expr or random expr")
            if self.add_zero_genes > 0 and self.how != "all":
                zero_loc = np.where(expr == 0)[0]
                zero_loc = [
                    np.random.choice(len(zero_loc), self.add_zero_genes, replace=False)
                ]
                loc = np.concatenate((loc, zero_loc), axis=None)
            exprs.append(expr[loc])
            gene_locs.append(loc + self.start_idx[organism_id])

            if self.tp_name is not None:
                tp.append(elem[self.tp_name])
            else:
                tp.append(0)

            other_classes.append([elem[i] for i in self.class_names])

        expr = np.array(exprs)
        tp = np.array(tp)
        gene_locs = np.array(gene_locs)
        total_count = np.array(total_count)
        other_classes = np.array(other_classes)

        # normalize counts
        if self.norm_to is not None:
            expr = (expr * self.norm_to) / total_count[:, None]
        if self.logp1:
            expr = np.log2(1 + expr)

        # do binning of counts
        if self.n_bins:
            pass

        # find the associated gene ids (given the species)

        # get the NN cells

        # do encoding / selection a la scGPT

        # do encoding of graph location
        # encode all the edges in some sparse way
        # normalizing total counts between 0,1
        return {
            "x": Tensor(expr),
            "genes": Tensor(gene_locs).int(),
            "class": Tensor(other_classes).int(),
            "tp": Tensor(tp),
            "depth": Tensor(total_count),
        }


class AnnDataCollator(Collator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, batch):
        exprs = []
        total_count = []
        other_classes = []
        gene_locs = []
        tp = []
        for elem in batch:
            organism_id = elem.obs[self.organism_name]
            if organism_id.item() not in self.organism_ids:
                print(organism_id)
            expr = np.array(elem.X[0])

            total_count.append(expr.sum())
            if len(self.accepted_genes) > 0:
                expr = expr[self.accepted_genes[organism_id]]
            if self.how == "most expr":
                loc = np.argsort(expr)[-(self.max_len) :][::-1]
            elif self.how == "random expr":
                nnz_loc = np.where(expr > 0)[0]
                loc = nnz_loc[
                    np.random.choice(len(nnz_loc), self.max_len, replace=False)
                ]
            else:
                raise ValueError("how must be either most expr or random expr")
            if self.add_zero_genes > 0:
                zero_loc = np.where(expr == 0)[0]
                zero_loc = [
                    np.random.choice(len(zero_loc), self.add_zero_genes, replace=False)
                ]
                loc = np.concatenate((loc, zero_loc), axis=None)
            exprs.append(expr[loc])
            gene_locs.append(loc + self.start_idx[organism_id.item()])

            if self.tp_name is not None:
                tp.append(elem.obs[self.tp_name])
            else:
                tp.append(0)

            other_classes.append([elem.obs[i].values[0] for i in self.class_names])

        expr = np.array(exprs)
        tp = np.array(tp)
        gene_locs = np.array(gene_locs)
        total_count = np.array(total_count)
        other_classes = np.array(other_classes)
        return {
            "x": Tensor(expr),
            "genes": Tensor(gene_locs).int(),
            "depth": Tensor(total_count),
            "class": Tensor(other_classes),
        }


class SCVICollator(Collator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, batch):
        expr = batch["x"]
        total_count = expr.sum(axis=1)
        if self.how == "most expr":
            loc = np.argsort(expr)[:, -(self.max_len) :][:, ::-1]
        else:
            raise ValueError("how must be either most expr or random expr")
        if self.logp1:
            expr = np.log2(1 + expr)
        return {
            "x": Tensor(expr[np.arange(expr.shape[0])[:, None], loc]),
            "genes": Tensor(loc.copy()).int(),
            "depth": Tensor(total_count),
        }


class GeneformerCollator(Collator):
    def __init__(self, *args, gene_norm_list: list, **kwargs):
        super().__init__(*args, **kwargs)
        self.gene_norm_list = gene_norm_list

    def __call__(self, batch):
        super().__call__(batch)
        # normlization per gene

        # tokenize the empty locations


class scGPTCollator(Collator):
    def __call__(self, batch):
        super().__call__(batch)
        # binning

        # tokenize the empty locations


class scPRINTCollator(Collator):
    def __call__(self, batch):
        super().__call__(batch)
