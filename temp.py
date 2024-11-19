import lamindb as ln
import tqdm
import anndata as ad
import pandas as pd
import scipy.sparse as sp
from scdataloader.utils import load_genes
import numpy as np

files = ln.Collection.filter(name="all").one().artifacts.filter()
genedf = load_genes(["NCBITaxon:9606", "NCBITaxon:10090"])
ok = {
 '2LDqOsp3fnwiTpU041lS',
 '380ZSAxJkq4dLZZEniAc',
 '6ldttob7GXaPuKP9kJAs',
 '7yVAt5M8JXSnBKFMp4Kv',
 '9JWbWWp3EPBpxolXq8ye',
 '9Z4F3X755K0BtrIgx9x0',
 '9z0raW4832qsWOZ7ktB5',
 'BRyGADLqmcuKreuUJ8Ji',
 'DgtXbidYsBmEQcrkvN4c',
 'DozflFCJKNBqXFm9BdqT',
 'EhCBr6389zshj0HccfDx',
 'ImySFGAfUL4MVSM0Kvfi',
 'JoiIpvCbZsgG8JCOEU2W',
 'MxHBVwQJhs9mfrAV3Ulk',
 'N6OgdojCnVh37L3r89gA',
 'OR4GQzmzUrrP2RAju9mw',
 'OYMOokbQDNNQCdY4PtF0',
 'PhLlT1Vvg7EOqCbc7JGy',
 'T0AHTHedTi73nQKIfeFF',
 'UUgLcOu5HPK5Tf6Otpei',
 'VDsUe1TLPT0Ird1MFuaV',
 'W85FniTKZjbEYND0Lz9Q',
 'XXDKnDJPrpjnr3dkXzsg',
 'Xx7mXup7coPbMhBwKsBp',
 'Y4MfxOK3tnmjEgKIoDbE',
 'Ynhb3z2vvcTJhqU1YJJO',
 'ZFC7I52xtDybEjyrUjan',
 'ajHnpjzN5pGinaUS9iB1',
 'd2QAIX2JJAfPtEYxE3nn',
 'dlUIY67zptLlpfvnMKOV',
 'e0dMKMTbhgN5CbYrVNyC',
 'gJGKwe9f0m4JPjPqAoYO',
 'hTZCvGvfgn214QwwOlFV',
 'jAJho1H8ohLeDWlVfggl',
 'jh8cK8LJ9k5j9XTMZOqU',
 'jlpIcfpQvfoyYrEUzfRv',
 'lA1xLOEaopmBe0XzrbWu',
 'm1pGsLjNOzggGgL5PaBB',
 'mSlsNUlgMUGub7VLpUcR',
 'sDy7km5rw4MlTGEodKBo',
 'snqqA3r5Bo17WEf9c4Mq',
 'xF91VRh1je3yMeeWDYFX',
 'zthHhrgKDGr6w8toNRfr',
 'zw5tvY9miK0QXaSRSeYq'}

for i, f in tqdm.tqdm(enumerate(files[:])):
    if f.uid in ok:
        print(i)
        adata = f.load()
        orga = adata.obs.organism_ontology_term_id.iloc[0]
        genedf_org = genedf[genedf.organism == orga]
        adata = adata[:, adata.var.index.isin(genedf_org.index)]
        unseen = set(genedf_org.index) - set(adata.var.index)
        # adding them to adata
        emptyda = ad.AnnData(
            sp.csr_matrix((adata.shape[0], len(unseen)), dtype=np.float32),
            var=pd.DataFrame(index=list(unseen)),
            obs=pd.DataFrame(index=adata.obs.index),
        )
        adata = ad.concat([adata, emptyda], axis=1, join="outer", merge="only")
        adata.write_h5ad(
            f"/pasteur/zeus/projets/p02/ml4ig_hot/Users/jkalfon/scprint/.lamindb/{f.uid}.h5ad"
        )
