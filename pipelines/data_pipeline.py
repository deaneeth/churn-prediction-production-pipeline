# This script orchestrates the data processing pipeline for a churn prediction model.
# It integrates various components such as data ingestion, missing value handling, outlier detection, feature binning, feature encoding, feature scaling, and data splitting.
# Each component is modular, allowing for easy adjustments and extensions as needed.
# simple explanation is that this script runs the entire data processing pipeline step by step and returns the final training and testing datasets as numpy arrays.

import os
import sys
import pandas as pd
from typing import Dict
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_ingestion import DataIngestorCSV
from handle_missing_values import DropMissingValuesStrategy, FillMissingValuesStrategy, GenderImputer, MissingValueHandlingStrategy
from outlier_detection import OutlierDetector, IQROutlierDetection
from feature_binning import FeatureBinningStrategy, CustomBinningStrategy
from feature_encoding import FeatureEncodingStrategy, NominalEncodingStrategy, OrdinalEncodingStrategy
from feature_scaling import FeatureScalingStrategy, MinMaxScalingStrategy
from data_spiltter import DataSplittingStrategy, SimpleTrainTestSplitStrategy
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config import get_data_paths, get_columns, get_missing_values_config, get_outlier_config, get_binning_config, get_encoding_config, get_scaling_config, get_splitting_config


# Main data pipeline function that executes all steps in sequence and returns the final datasets as numpy arrays 
def data_pipeline(
                    data_path: str='data/raw/ChurnModelling.csv',   # Path to the raw data file
                    target_column: str='Exited',                    # Target variable for prediction
                    test_size: float=0.2,                           # Proportion of data to be used for testing
                    force_rebuild: bool=False                       # Flag to force reprocessing of data even if processed files exist
                    ) -> Dict[str, np.ndarray]:                     # Returns a dictionary containing training and testing datasets as numpy arrays
    
    # Load configurations for various steps in the pipeline from the config module 
    data_paths = get_data_paths()                                # Get paths for data artifacts
    columns = get_columns()                                      # Get column names for various steps
    missing_values_config = get_missing_values_config()          # Get configuration for handling missing values
    outlier_config = get_outlier_config()                        # Get configuration for outlier detection
    binning_config = get_binning_config()                        # Get configuration for feature binning
    encoding_config = get_encoding_config()                      # Get configuration for feature encoding
    scaling_config = get_scaling_config()                        # Get configuration for feature scaling
    splitting_config = get_splitting_config()                    # Get configuration for data splitting
    
    
    
    # Step 1: Data Ingestion 
    print('\nStep 1: Data Ingestion')
    artifacts_dir = os.path.join(os.path.dirname(__file__), '..', data_paths['data_artifacts_dir'])  # Directory to save processed data artifacts
    x_train_path = os.path.join(artifacts_dir, 'X_train.csv')     # Path to save training features
    x_test_path = os.path.join(artifacts_dir, 'X_test.csv')       # Path to save testing features
    y_train_path = os.path.join(artifacts_dir, 'Y_train.csv')     # Path to save training labels
    y_test_path = os.path.join(artifacts_dir, 'Y_test.csv')       # Path to save testing labels
    
    # If only the final processed files exist and force_rebuild is False, load them directly and skip processing steps 
    if  os.path.exists(x_train_path) and \
        os.path.exists(x_test_path) and \
        os.path.exists(y_train_path) and \
        os.path.exists(y_test_path) and not force_rebuild:
            
            # Load training features from CSV
            X_train = pd.read_csv(x_train_path)     
            X_test = pd.read_csv(x_test_path)
            Y_train = pd.read_csv(y_train_path)
            Y_test = pd.read_csv(y_test_path)
            print('Processed files found. Loaded training and testing datasets directly from artifacts.')
            
    os.makedirs(data_paths['data_artifacts_dir'], exist_ok=True)
    
    
# now we are calling all the classes and functions from the other files to run the entire data processing pipeline step by step. 

    
    # Step 2 : Handle missing values based on the specified strategy in the configuration
    print('\nStep 2: Handle Missing Values')
        
    # If processed files do not exist or force_rebuild is True, run the full data processing pipeline
    if not os.path.exists('data/imputed/temp_imputed.csv'):              # If temp_imputed.csv does not exist, we need to run the full pipeline
        print("Processed files not found for Step 2. Running data processing pipeline...")
    
        # run the full data processing pipeline
        ingestor = DataIngestorCSV()                                  # Initialize data ingestor
        df = ingestor.ingest(data_path)                               # Ingest raw data
        print(f'Loaded data shape: {df.shape}')
        
        drop_handler = DropMissingValuesStrategy(critical_columns = columns['critical_columns'])   # you can take critical columns from columns (drop some rows based on critical columns)
        
        # Debug to check if the 'Age' column exists in the dataframe
        print(f"Columns in dataframe: {df.columns.tolist()}")
        
        
        # Corrected initialization of age_handler with relevant_column parameter
        age_handler = FillMissingValuesStrategy(
                                                method = 'mean', 
                                                relevant_column = 'Age'
                                                )
        
        # Custom imputer 
        gender_handler = FillMissingValuesStrategy(
                                                        relevant_column = 'Gender',
                                                        is_custom_imputer = True,
                                                        custom_imputer = GenderImputer()
                                                    )
        
        
        df = drop_handler.handle(df)                      # Drop rows with missing values in critical columns
        df = age_handler.handle(df)                       # Impute missing values in 'Age' column
        df = gender_handler.handle(df)                    # Impute missing values
        df.to_csv('data/imputed/temp_imputed.csv', index=False)        # Save intermediate result (.csv) after handling missing values
        print("Missing values handled and intermediate result saved to 'data/imputed/temp_imputed.csv'.")

    df = pd.read_csv('data/imputed/temp_imputed.csv')                   # Load the intermediate result for further processing from the saved .csv file
    print('Processed files found. Loading imputed data from temp_imputed.csv')
    print(f"Data Shape after imputation : {df.shape}")
    
    
    # Step 3: Outlier Detection and Removal
    print('\nStep 3: Outlier Detection and Removal')
    
    outlier_detector = OutlierDetector(strategy=IQROutlierDetection())                     # Initialize outlier detector with IQR strategy
    df = outlier_detector.handle_outliers(df, columns['outlier_columns'])                  # Remove rows with outliers in specified columns
    
    print(f"Data Shape after outlier removal : {df.shape}")
    
    
    # Step 4: Feature Binning for 'CreditScore' column using custom binning strategy 
    print('\nStep 4: Feature Binning')
    
    binning = CustomBinningStrategy(binning_config['credit_score_bins'])              # Initialize custom binning strategy with defined bins
    df = binning.bin_feature(df, 'CreditScore')                                       # Apply binning to 'CreditScore' column
    print(f"Data after feature binning: \n{df.head()}")                
    
    # Step 5: Feature Encoding for categorical variables using nominal and ordinal encoding strategies
    print('\nStep 5: Feature Encoding')
    
    nominal_encoding_strategy = NominalEncodingStrategy(encoding_config['nominal_columns'])     # Initialize nominal encoding strategy with specified columns
    ordinal_encoding_strategy = OrdinalEncodingStrategy(encoding_config['ordinal_mappings'])    # Initialize ordinal encoding strategy with specified mappings
    
    df = nominal_encoding_strategy.encode(df)           # Apply nominal encoding
    df = ordinal_encoding_strategy.encode(df)           # Apply ordinal encoding
    print(f"Data after feature encoding: \n{df.head()}")
    
    
    # Step 6: Feature Scaling for numerical variables using Min-Max scaling strategy
    print('\nStep 6: Feature Scaling')
    
    minmax_strategy = MinMaxScalingStrategy()                                           # Initialize Min-Max scaling strategy
    df = minmax_strategy.scale(df, scaling_config['columns_to_scale'])                  # Apply Min-Max scaling to specified columns
    print(f"Data after feature scaling: \n{df.head()}")
    
    
    # Step 7: Post Processing - Drop unnecessary columns before final data splitting
    print('\nStep 7: Post Processing')
    
    df = df.drop(columns=['RowNumber', 'CustomerId', 'Firstname', 'Lastname'])           # Drop unnecessary columns
    print(f"Data after post processing: \n{df.head()}")
    
    
    # Step 8: Data Splitting into training and testing sets based on the specified strategy in the configuration
    print('\nStep 8: Data Splitting')
    
    splitting_strategy = SimpleTrainTestSplitStrategy(test_size=0.3)                 # Initialize data splitting strategy with specified test size
    X_train, X_test, Y_train, Y_test = splitting_strategy.split_data(df, 'Exited')   # Split data into training and testing sets based on the target column
    
    X_train.to_csv(x_train_path, index=False)       
    X_test.to_csv(x_test_path, index=False)
    Y_train.to_csv(y_train_path, index=False)
    Y_test.to_csv(y_test_path, index=False)
    
    print(f"X train size : {X_train.shape}")
    print(f"X test size : {X_test.shape}")
    print(f"Y train size : {Y_train.shape}")
    print(f"Y test size : {Y_test.shape}")
    
    
data_pipeline() # Call the data pipeline function to execute the entire process