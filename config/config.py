from dataclasses import dataclass
from typing import Dict


@dataclass
class Config:
    # Data
    description: str
    reinit_lamindb: str = True
    start_at: int = 0
    version: int = 1

    additional_tissues: Dict = {
        "UBERON:0037144": "wall of heart",
        "UBERON:0003929": "digestive tract epithelium",
        "UBERON:0002020": "gray matter",
        "UBERON:0000200": "gyrus",
        "UBERON:0000101": "lobe of lung",
        "UBERON:0001981": "blood vessel",
        "UBERON:0001474": "bone element",
    }

    additional_diseases: Dict = {
        "MONDO:0001106": "kidney failure",
        "MONDO:0021166": "inflammatory disease",
        "MONDO:0004992": "cancer",
        "MONDO:0004994": "cardiomyopathy",
        "MONDO:0700065": "trisomy",
        "MONDO:0021042": "glioma",
        "MONDO:0005265": "inflammatory bowel disease",
        "MONDO:0005550": "infectious disease",
        "MONDO:0005059": "leukemia",
    }

    additional_assays: Dict = {
        "EFO:0010184": "Smart-like",
        "EFO:0010961": "Visium Spatial Gene Expression",
    }
