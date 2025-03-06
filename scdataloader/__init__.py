from importlib.metadata import version

from .collator import Collator
from .data import Dataset, SimpleAnnDataset
from .datamodule import DataModule
from .preprocess import Preprocessor

__version__ = version("scdataloader")
