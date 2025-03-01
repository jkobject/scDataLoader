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


MAIN_HUMAN_MOUSE_DEV_STAGE_MAP = {
    "HsapDv:0010000": [
        "MmusDv:0000092",  # postnatal stage
    ],
    "HsapDv:0000258": [  # mature stage
        "MmusDv:0000110",  # mature stage
        "HsapDv:0000204", # 
    ],
    "HsapDv:0000227": [  # late adult stage
        "MmusDv:0000091",  # 20 month-old stage
        "MmusDv:0000089",  # 18 month-old stage
    ],
    "HsapDv:0000272": [],  # 60-79 year-old stage
    "HsapDv:0000095": [],  # 80 year-old and over stage
    "HsapDv:0000267": [  # middle aged stage
        "MmusDv:0000087",  # 16 month-old stage
        "UBERON:0018241",  # prime adult stage
        "MmusDv:0000083",  # 12 month-old stage
        "HsapDv:0000092",  # same
    ],
    "HsapDv:0000266": [  # young adult stage
        "MmusDv:0000050",  # 6 weeks
        "HsapDv:0000089",  # same
        "MmusDv:0000051",  # 7 weeks
        "MmusDv:0000052",  # 8 weeks
        "MmusDv:0000053",  # 9 weeks
        "MmusDv:0000054",  # 10 weeks
        "MmusDv:0000055",  # 11 weeks
        "MmusDv:0000056",  # 12 weeks
        "MmusDv:0000057",  # 13 weeks
        "MmusDv:0000058",  # 14 weeks
        "MmusDv:0000059",  # 15 weeks
        "MmusDv:0000061",  # early adult stage
        "MmusDv:0000062",  # 2 month-old stage
        "MmusDv:0000063",  # 3 month-old stage
        "MmusDv:0000064",  # 4 month-old stage
        "MmusDv:0000065",  # 16 weeks
        "MmusDv:0000066",  # 17 weeks
        "MmusDv:0000067",  # 18 weeks
        "MmusDv:0000068",  # 19 weeks
        "MmusDv:0000070",  # 20 weeks
        "MmusDv:0000071",  # 21 weeks
        "MmusDv:0000072",  # 22 weeks
        "MmusDv:0000073",  # 23 weeks
        "MmusDv:0000074",  # 24 weeks
        "MmusDv:0000077",  # 6 month-old stage
        "MmusDv:0000079",  # 8 month-old stage
        "MmusDv:0000098",  # 25 weeks
        "MmusDv:0000099",  # 26 weeks
        "MmusDv:0000102",  # 29 weeks
    ],
    "HsapDv:0000265": [],  # child stage (1-4 yo)
    "HsapDv:0000271": [  # juvenile stage (5-14 yo)
        "MmusDv:0000048",  # 4 weeks
        "MmusDv:0000049",  # 5 weeks
    ],
    "HsapDv:0000260": [  # infant stage
        "MmusDv:0000046",  # 2 weeks
        "MmusDv:0000045",  # 1 week
        "MmusDv:0000047",  # 3 weeks
        "HsapDv:0000083",
    ],
    "HsapDv:0000262": [  # newborn stage (0-28 days)
        "MmusDv:0000036",  # Theiler stage 27
        "MmusDv:0000037",  # Theiler stage 28
        "MmusDv:0000113",  # 4-7 days
    ],
    "HsapDv:0000007": [],  # Carnegie stage 03
    "HsapDv:0000008": [],  # Carnegie stage 04
    "HsapDv:0000009": [],  # Carnegie stage 05
    "HsapDv:0000003": [],  # Carnegie stage 01
    "HsapDv:0000005": [],  # Carnegie stage 02
    "HsapDv:0000010": [],  # gastrula stage
    "HsapDv:0000012": [],  # neurula stage
    "HsapDv:0000015": [  # organogenesis stage
        "MmusDv:0000019",  # Theiler stage 13
        "MmusDv:0000020",  # Theiler stage 12
        "MmusDv:0000021",  # Theiler stage 14
        "MmusDv:0000022",  # Theiler stage 15
        "MmusDv:0000023",  # Theiler stage 16
        "MmusDv:0000024",  # Theiler stage 17
        "MmusDv:0000025",  # Theiler stage 18
        "MmusDv:0000026",  # Theiler stage 19
        "MmusDv:0000027",  # Theiler stage 20
        "MmusDv:0000028",  # Theiler stage 21
        "MmusDv:0000029",  # Theiler stage 22
    ],
    "HsapDv:0000037": [  # fetal stage
        "MmusDv:0000033",  # Theiler stage 24
        "MmusDv:0000034",  # Theiler stage 25
        "MmusDv:0000035",  # Theiler stage 26
        "MmusDv:0000032",  # Theiler stage 23
    ],
    "unknown": [
        "MmusDv:0000041",  # unknown
    ],
}
