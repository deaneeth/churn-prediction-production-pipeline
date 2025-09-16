import logging
import pandas as pd
from enum import Enum
from abc import ABC, abstractmethod
from typing import Tuple
from sklearn.model_selection import train_test_split
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')


class DataSplittingStrategy(ABC):
    @abstractmethod
    def split_data(self, df: pd.DataFrame, target_column: str) ->Tuple[pd.
        DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        pass


class SplitType(str, Enum):
    SIMPLE = 'simple' # Simple Splitter I utilized
    STRATIFIED = 'stratified'

