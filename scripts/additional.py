import scanpy as sc
import pandas as pd

https://cells.ucsc.edu/hoc/blood/exprMatrix.tsv.gz
https://cells.ucsc.edu/hoc/blood/meta.tsv


curl -O https://cf.10xgenomics.com/samples/cell-exp/1.3.0/1M_neurons/1M_neurons_filtered_gene_bc_matrices_h5.h5
curl -O https://cf.10xgenomics.com/samples/cell-exp/1.3.0/1M_neurons/1M_neurons_reanalyze.csv


https://ftp.ncbi.nlm.nih.gov/geo/series/GSE247nnn/GSE247719/suppl/GSE247719%5F20240213%5FPanSci%5Fall%5Fcells%5Fadata.h5ad.gz

ad = sc.read_mtx("matrix.mtx.gz")
meta = pd.read_csv("meta.tsv", sep="\t")
ad.var = meta

# 95G     /home/ml4ig1/.cache/lamindb/cellxgene-data-public/cell-census/2023-12-15/h5ads/
# 40G     /home/ml4ig1/scprint/cell-census/2023-07-25/h5ads/
# 197G    /home/ml4ig1/scprint/.lamindb/
# /home/ml4ig1/Documents code/scGPT/mytests/attn_scores_l11.pkl 8G


/home/ml4ig1/Documents code/scPRINT