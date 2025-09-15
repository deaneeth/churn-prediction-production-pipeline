# This file is responsible for defining various feature scaling strategies for numerical variables.
# It includes an implementation for Min-Max scaling.
# The module is designed to be flexible, allowing for easy addition of new scaling strategies as needed
# It integrates with the overall data processing pipeline to ensure that numerical features are appropriately scaled for model training.


import logging
import pandas as pd
from enum import Enum
from typing import List
from abc import ABC, abstractmethod
from sklearn.preprocessing import MinMaxScaler
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')


# Define the abstract base class for feature scaling strategies
class FeatureScalingStrategy(ABC):

    @abstractmethod
    def scale(self, df: pd.DataFrame, columns_to_scale: List[str]) -> pd.DataFrame:
        pass

# Concrete implementation for Min-Max Scaling
class ScalingType(str, Enum):
    MINMAX = 'minmax'
    STANDARD = 'standard'

# Min-Max Scaling Strategy
class MinMaxScalingStrategy(FeatureScalingStrategy):
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.fitted = False

    def scale(self, df, columns_to_scale):
        df[columns_to_scale] = self.scaler.fit_transform(df[columns_to_scale])   # Fit and transform the specified columns
        self.fitted = True
        logging.info(f'Applied Min-Max scaling to columns: {columns_to_scale}')  # log the scaling action
        return df 
    
    def get_scaler(self):       # method to retrieve the fitted scaler
        return self.scaler      # return the fitted scaler
    
    
    
# This scaler can be extended with additional strategies like StandardScaler as needed.
