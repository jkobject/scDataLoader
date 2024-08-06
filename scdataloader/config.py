"""
Configuration file for scDataLoader

Missing labels are added to the dataset to complete a better hierarchical tree
"""

LABELS_TOADD = {
    "assay_ontology_term_id": {
        "10x transcription profiling": "EFO:0030003",
        "spatial transcriptomics": "EFO:0008994",
        "10x 3' transcription profiling": "EFO:0030003",
        "10x 5' transcription profiling": "EFO:0030004",
    },
    "disease_ontology_term_id": {
        "metabolic disease": "MONDO:0005066",
        "chronic kidney disease": "MONDO:0005300",
        "chromosomal disorder": "MONDO:0019040",
        "infectious disease": "MONDO:0005550",
        "inflammatory disease": "MONDO:0021166",
        # "immune system disease",
        "disorder of development or morphogenesis": "MONDO:0021147",
        "mitochondrial disease": "MONDO:0044970",
        "psychiatric disorder": "MONDO:0002025",
        "cancer or benign tumor": "MONDO:0002025",
        "neoplasm": "MONDO:0005070",
    },
    "cell_type_ontology_term_id": {
        "progenitor cell": "CL:0011026",
        "hematopoietic cell": "CL:0000988",
        "myoblast": "CL:0000056",
        "myeloid cell": "CL:0000763",
        "neuron": "CL:0000540",
        "electrically active cell": "CL:0000211",
        "epithelial cell": "CL:0000066",
        "secretory cell": "CL:0000151",
        "stem cell": "CL:0000034",
        "non-terminally differentiated cell": "CL:0000055",
        "supporting cell": "CL:0000630",
    },
}

COARSE_TISSUE = {
    "adipose tissue": "",
    "bladder organ": "",
    "blood": "",
    "bone marrow": "",
    "brain": "",
    "breast": "",
    "esophagus": "",
    "eye": "",
    "embryo": "",
    "fallopian tube": "",
    "gall bladder": "",
    "heart": "",
    "intestine": "",
    "kidney": "",
    "liver": "",
    "lung": "",
    "lymph node": "",
    "musculature of body": "",
    "nose": "",
    "ovary": "",
    "pancreas": "",
    "placenta": "",
    "skin of body": "",
    "spinal cord": "",
    "spleen": "",
    "stomach": "",
    "thymus": "",
    "thyroid gland": "",
    "tongue": "",
    "uterus": "",
}

COARSE_ANCESTRY = {
    "African": "",
    "Chinese": "",
    "East Asian": "",
    "Eskimo": "",
    "European": "",
    "Greater Middle Eastern  (Middle Eastern, North African or Persian)": "",
    "Hispanic or Latin American": "",
    "Native American": "",
    "Oceanian": "",
    "South Asian": "",
}

COARSE_DEVELOPMENT_STAGE = {
    "Embryonic human": "",
    "Fetal": "",
    "Immature": "",
    "Mature": "",
}

COARSE_ASSAY = {
    "10x 3'": "",
    "10x 5'": "",
    "10x multiome": "",
    "CEL-seq2": "",
    "Drop-seq": "",
    "GEXSCOPE technology": "",
    "inDrop": "",
    "microwell-seq": "",
    "sci-Plex": "",
    "sci-RNA-seq": "",
    "Seq-Well": "",
    "Slide-seq": "",
    "Smart-seq": "",
    "SPLiT-seq": "",
    "TruDrop": "",
    "Visium Spatial Gene Expression": "",
}
