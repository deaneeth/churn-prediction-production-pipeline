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

# Nominal Encoding Strategy
class NominalEncodingStrategy(FeatureEncodingStrategy):
    def __init__(self, nominal_columns):
            self.nominal_columns = nominal_columns    # creating a variable here 
            self.encoder_dicts = {}    
            
            os.makedirs('artifacts/encode', exist_ok=True)   # create directory if not exists
            
    def encode(self, df):
        
        for column in self.nominal_columns:
            unique_values = df[column].unique()
            encoder_dict = {value: i for i, value in enumerate(unique_values)}          # create a mapping dictionary
            self.encoder_dicts[column] = encoder_dict
            
            encoder_path = os.path.join('artifacts/encode', f"{column}_encoder.json")   # define the path to save the encoder
            with open(encoder_path, "w") as f:
                json.dump(encoder_dict, f)                                              # Fixed: Added file object as second parameter
                
            df[column] = df[column].map(encoder_dict)      # map the original values to their encoded integers
        
        return df
    
    def get_encoder_dicts(self):            # method to retrieve the encoder dictionaries
        return self.encoder_dicts
    
    
# Ordinal Encoding Strategy
class OrdinalEncodingStrategy(FeatureEncodingStrategy):
    def __init__(self, ordinal_mappings):
            self.ordinal_mappings = ordinal_mappings    # creating a variable here
            
    def encode(self, df):     
        for column, mapping in self.ordinal_mappings.items():                                      # iterate through the mapping dictionary
            df[column] = df[column].map(mapping)
            logging.info(f"Encoded ordinal variable '{column}' with {len(mapping)} categories")    # log the encoding process
            
        return df                                                                                  # Fixed: Moved outside the loop to return after processing all columns
    
    
    
# This way, we can easily add new encoding strategies by creating new classes that inherit from FeatureEncodingStrategy and implement the encode method.
# We can then integrate these strategies into our data processing pipeline to ensure that categorical features are appropriately transformed for model training.
