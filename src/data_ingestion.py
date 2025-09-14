# Data Ingestion Module
# This module is responsible for ingesting data from various sources.

import os
import pandas as pd
from abc import ABC, abstractmethod # abstract base class for creating abstract classes and methods


class DataIngestor(ABC):        # the idea is that if you are calling this data ingestor class, you must have this ingest method
    @abstractmethod
    def ingest(self, file_path_or_link: str) ->pd.DataFrame:
        pass
    
# Concrete implementation for CSV files
class DataIngestorCSV(DataIngestor):   # all the carasteristics that DataIngestor class have should followed by the DataIngestorCSV class
    def ingest(self, file_path_or_link):
        return pd.read_csv(file_path_or_link)

# Concrete implementation for Excel files
class DataIngestorExcel(DataIngestor):
    def ingest(self, file_path_or_link):
        return pd.read_excel(file_path_or_link)
        
    
# Now we loaded the data using the appropriate ingestor, next handle missing values