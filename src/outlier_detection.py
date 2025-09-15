# This file is responsible for detecting and handling outliers in the dataset.
# It uses the Interquartile Range (IQR) method to identify outliers and provides options to either remove or cap them.
# The module is designed to be flexible, allowing for different strategies to be implemented as needed.
# It integrates with the overall data processing pipeline to ensure data quality before model training.

import logging
import pandas as pd
from abc import ABC, abstractmethod
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')

class OutlierDetectionStrategy(ABC):                     # the class that inherited by 'Abstract base class' is the 'OutlierDetectionStrategy'
    @abstractmethod
    def detect_outliers(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:   # abstract method that must be implemented by any subclass 
        pass

# Concrete implementation for IQR method 
class IQROutlierDetection(OutlierDetectionStrategy):
    def detect_outliers(self, df, columns):
        outliers = pd.DataFrame(False, index=df.index, columns=columns)  # DataFrame to store outlier flags 
        
        for col in columns:
            Q1 = df[col].quantile(0.25)    # First quartile (25th percentile)
            Q3 = df[col].quantile(0.75)    # Third quartile (75th percentile)
            IQR = Q3 - Q1                  # Interquartile range
            outliers[col] = (df[col] < Q1 - 1.5 * IQR | df[col] > Q3 + 1.5 * IQR)   # Identify outliers 
            
        logging.info("Outlier detected using IQR method.")
        return outliers

# Outlier Detector Class
class OutlierDetector:
    def __init__(self, strategy):                # strategy is an instance of a class that implements OutlierDetectionStrategy
        self._strategy = strategy                # assign the strategy to a private attribute
        
    def detect_outliers(self, df, selected_columns ):
        return self._strategy.detect_outliers(df, selected_columns)        # delegate the outlier detection to the strategy
    
    def handle_outliers(self, df, selected_columns, method='remove'):     
        outliers = self.detect_outliers(df, selected_columns)
        outlier_count = outliers.sum(axis=1)                               # count of outliers per row
        rows_to_remove = outlier_count >= 2                                # rows with outliers in 2 or more columns
        
        return df[~rows_to_remove]                                         # return DataFrame with outlier rows removed 
 
 
# Now we can integrate this OutlierDetector into our data processing pipeline to ensure data quality before model training.