# This is the file where we loading the config.yaml file and read the parameters from it.
# We will create functions to read different sections of the config file.
# We will also create a function to update the config file if needed.
# We will also create a function to create a default config file if it does not exist.

import os
import yaml  #to load the data from config,yaml file you need to use pyaml
import logging
from typing import Dict, Any, List
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)),
    'config.yaml')

# Function to load the config file
def load_config():
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f'Error loading configuration: {e}')
        return {}


def get_data_paths():
    config = load_config()
    return config.get('data_paths', {})


def get_columns():
    config = load_config()
    return config.get('columns', {})


def get_missing_values_config():
    config = load_config()
    return config.get('missing_values', {})


def get_outlier_config():
    config = load_config()
    return config.get('outlier_detection', {})


def get_binning_config():
    config = load_config()
    return config.get('feature_binning', {})


def get_encoding_config():
    config = load_config()
    return config.get('feature_encoding', {})


def get_scaling_config():
    config = load_config()
    return config.get('feature_scaling', {})


def get_splitting_config():
    config = load_config()
    return config.get('data_splitting', {})


def get_training_config():
    config = load_config()
    return config.get('training', {})


def get_model_config():
    config = load_config()
    return config.get('model', {})


def get_evaluation_config():
    config = load_config()
    return config.get('evaluation', {})


def get_deployment_config():
    config = load_config()
    return config.get('deployment', {})


def get_logging_config():
    config = load_config()
    return config.get('logging', {})


def get_environment_config():
    config = load_config()
    return config.get('environment', {})


def get_pipeline_config():
    config = load_config()
    return config.get('pipeline', {})


def get_inference_config():
    config = load_config()
    return config.get('inference', {})


def get_config() ->Dict[str, Any]:
    return load_config()


def get_data_config() ->Dict[str, Any]:
    config = get_config()
    return config.get('data', {})


def get_preprocessing_config() ->Dict[str, Any]:
    config = get_config()
    return config.get('preprocessing', {})


# Get the selected model configuration
def get_selected_model_config() ->Dict[str, Any]:
    training_config = get_training_config()
    selected_model = training_config.get('selected_model', 'random_forest')
    model_types = training_config.get('model_types', {})
    return {'model_type': selected_model, 'model_config': model_types.get(
        selected_model, {}), 'training_strategy': training_config.get(
        'training_strategy', 'cv'), 'cv_folds': training_config.get(
        'cv_folds', 5), 'random_state': training_config.get('random_state', 42)
        }


# Get a list of all available model types from the config file
def get_available_models() ->List[str]:
    training_config = get_training_config()
    return list(training_config.get('model_types', {}).keys())


# Function to update the config file with new values
def update_config(updates: Dict[str, Any]) ->None:
    config_path = CONFIG_FILE
    config = get_config()
    for key, value in updates.items():
        keys = key.split('.')
        current = config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
    with open(config_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)


# Function to create a default config file if it does not exist
def create_default_config() ->None:
    config_path = CONFIG_FILE
    if not os.path.exists(config_path):
        default_config = {'data': {'file_path':
            'data/raw/ChurnModelling.csv', 'target_column': 'Exited',
            'test_size': 0.2, 'random_state': 42}, 'preprocessing': {
            'handle_missing_values': True, 'handle_outliers': True,
            'feature_binning': True, 'feature_encoding': True,
            'feature_scaling': True}, 'training': {'selected_model':
            'random_forest', 'training_strategy': 'cv', 'cv_folds': 5,
            'random_state': 42}}
        with open(config_path, 'w') as file:
            yaml.dump(default_config, file, default_flow_style=False)
        logger.info(f'Created default configuration file: {config_path}')


create_default_config()
