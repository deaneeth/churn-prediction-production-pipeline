# This file is responsible for defining various feature binning strategies for numerical variables.
# It includes implementations for equal-width and equal-frequency binning.
# The module is designed to be flexible, allowing for easy addition of new binning strategies as needed.
# It integrates with the overall data processing pipeline to ensure that numerical features are appropriately transformed for model training.


import logging
import pandas as pd
from abc import ABC, abstractmethod
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')


# Abstract Base Class for Feature Binning Strategies
class FeatureBinningStrategy(ABC):
    @abstractmethod
    def bin_feature(self, df: pd.DataFrame, column: str) ->pd.DataFrame:
        pass
    
# custom creditscore binning functions
class CustomBinningStrategy(FeatureBinningStrategy):
    def __init__(self, bin_definitions):
        self.bin_definitions = bin_definitions
        
    def bin_feature(self, df, column):                  # implement the abstract method 'bin_feature'
            def assign_bin(value):
                if value == 850:
                    return "Excellent"                  # special case for 850


                for bin_label, bin_range in self.bin_definitions.items():     # iterate through the bin definitions
                    if len(bin_range) == 2:
                        if bin_range[0] <= value <= bin_range[1]:             # check if value falls within the range
                            return bin_label
                        
                    elif len(bin_range) == 1:                                 # single boundary condition
                        if value >= bin_range[0]:                             # check if value is greater than or equal to the boundary
                            return bin_label
                        
                if value > 850:                                               # handle values greater than 850
                    return "Invalid"   
                
                return "Invalid"
            
            df[f'{column}Bins'] = df[column].apply(assign_bin)                # create a new column with binned values
            del df[column]
            
            return df