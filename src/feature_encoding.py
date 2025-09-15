# This file is responsible for defining various feature encoding strategies for categorical variables.
# It includes implementations for nominal and ordinal encoding.
# The module is designed to be flexible, allowing for easy addition of new encoding strategies as needed
# It integrates with the overall data processing pipeline to ensure that categorical features are appropriately transformed for model training.

import logging
import pandas as pd
import os
import json
from enum import Enum
from typing import Dict, List
from abc import ABC, abstractmethod
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')


class FeatureEncodingStrategy(ABC):                # the class that inherited by 'Abstract base class' is the 'FeatureEncodingStrategy'
    @abstractmethod
    def encode(self, df: pd.DataFrame) ->pd.DataFrame:  
        pass

class VariableType(str, Enum):     # Define variable types for encoding strategies
    NOMINAL = 'nominal'
    ORDINAL = 'ordinal'