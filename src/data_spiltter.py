# This module provides functionality to split a dataset into training and testing sets.
# It includes an implementation for a simple random split.
# The module is designed to be flexible, allowing for easy addition of new splitting strategies as needed
# It integrates with the overall data processing pipeline to ensure that data is appropriately split for model training

import logging
import pandas as pd
from enum import Enum
from abc import ABC, abstractmethod
from typing import Tuple
from sklearn.model_selection import train_test_split
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')


class DataSplittingStrategy(ABC):                       # Abstract base class for data splitting strategies
    @abstractmethod
    def split_data(self, df: pd.DataFrame, target_column: str) ->Tuple[pd.       
        DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        pass


# Concrete implementation for simple train-test split
class SplitType(str, Enum):
    SIMPLE = 'simple'               # Simple Splitter I'm utilizing here
    STRATIFIED = 'stratified'


# Simple Train-Test Split Strategy
class SimpleTrainTestSplitStrategy(DataSplittingStrategy):
    def __init__(self, test_size = 0.2):
        self.test_size = test_size
        
    def split_data(self, df, target_column):                  # Split the data into features and target
        Y = df[target_column]                                 # Extract target variable
        X = df.drop(columns=[target_column])                  # Extract feature variables

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=self.test_size)              # Perform the train-test split
        logging.info(f'Split data into train and test sets with test size = {self.test_size}')  # Log the splitting action
        return X_train, X_test, Y_train, Y_test                                                 # Return the split datasets
    
    
# This splitter can be extended with additional strategies like StratifiedSplitter as needed.
# The difference between simple and stratified is that simple randomly splits the data into train and test sets, while stratified ensures that the proportion of classes in the target variable is maintained in both the train and test sets.
