import logging
import pandas as pd
from abc import ABC, abstractmethod
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')


class FeatureBinningStrategy(ABC):
    @abstractmethod
    def bin_feature(self, df: pd.DataFrame, column: str) ->pd.DataFrame:
        pass

class CustomBinningStratergy(FeatureBinningStrategy):
    def __init__(self, bin_definitions):
        self.bin_definitions = bin_definitions 

    def bin_feature(self, df, column):
        def assign_bin(value):
            if value == 850:
                return "Excellent"
            
            for bin_label, bin_range in self.bin_definitions.items():
                if len(bin_range) == 2:
                    if bin_range[0] <= value <= bin_range[1]:
                        return bin_label
                elif len(bin_range) == 1:
                    if value >= bin_range[0]:
                        return bin_label 
                
            if value > 850:
                return "Invalid"
            
            return "Invalid"
        
        df[f'{column}Bins'] = df[column].apply(assign_bin)
        del df[column]

        return df