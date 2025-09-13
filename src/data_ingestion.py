# Data Ingestion Module
# This module is responsible for ingesting data from various sources.

import os
import pandas as pd
from abc import ABC, abstractmethod # abstract base class for creating abstract classes and methods


class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path_or_link: str) ->pd.DataFrame:
        pass
    
    
    
    