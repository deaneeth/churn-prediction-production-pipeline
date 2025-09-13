# 🚀 Customer Churn Prediction Production Pipeline

![Churn Prediction](https://img.shields.io/badge/ML-Churn%20Prediction-blue)
![Python](https://img.shields.io/badge/Python-3.11%2B-brightgreen)
![Status](https://img.shields.io/badge/Status-Production--Ready-success)

Welcome to the production pipeline phase of the Customer Churn Prediction project! This is the third repository in a series focused on building a complete churn prediction system. It follows the work from [Customer Churn Prediction – EDA & Data Preprocessing Pipeline](https://github.com/deaneeth/customer-churn-eda-preprocessing) and [Customer Churn Prediction – Model Training & Evaluation Pipeline](https://github.com/deaneeth/customer-churn-model-training).

This repository brings together all the learnings from previous phases to create a comprehensive end-to-end machine learning pipeline for customer churn prediction, designed with production deployment in mind. It implements a robust ML workflow from data ingestion to model deployment and streaming inference.

## 📋 Table of Contents

- [Overview](#-overview)
- [Steps Followed from Previous Repositories](#-steps-followed-from-the-previous-repositories)
- [Project Structure](#-project-structure)
- [Features](#-features)
- [Requirements](#-requirements)
- [Installation](#️-installation)
- [Usage](#-usage)
- [Pipeline Components](#-pipeline-components)
- [Configuration](#️-configuration)
- [Model Performance](#-model-performance)
- [Deployment](#-deployment)
- [Contributing](#-contributing)
- [License](#-license)

## 🔍 Overview

This project provides a production-ready pipeline for predicting customer churn based on historical data. It includes comprehensive data preprocessing, feature engineering, model training, evaluation, and inference capabilities. The modular architecture allows for easy maintenance, scaling, and adaptation to similar prediction problems.

## 📌 Steps Followed from the Previous Repositories

If you're new to this series, it's recommended to explore the previous repositories first:

1. [Customer Churn Prediction – EDA & Data Preprocessing Pipeline](https://github.com/deaneeth/customer-churn-eda-preprocessing) - Focuses on exploratory data analysis and preprocessing steps, including handling missing values, encoding features, and scaling the dataset, which are essential steps before model training.

2. [Customer Churn Prediction – Model Training & Evaluation Pipeline](https://github.com/deaneeth/customer-churn-model-training) - Builds on the preprocessing work to train machine learning models, evaluate their performance, and save the trained models for future use.

This repository combines the learnings from both previous phases and adds production-level architecture for deployment-ready inference capabilities. Note that this is part of an ongoing series, with more advanced implementations planned for future repositories.

## 📁 Project Structure

```bash
churn-prediction-production-pipeline/
├── artifacts/               # Model artifacts and processed data
│   ├── data/                # Split datasets
│   └── encode/              # Encoding artifacts
├── data/                    # Data directory
│   └── raw/                 # Raw dataset
├── pipelines/               # End-to-end pipelines
│   ├── data_pipeline.py     # Data preprocessing pipeline
│   ├── training_pipeline.py # Model training pipeline
│   └── streaming_inference_pipeline.py # Inference pipeline
├── src/                     # Core functionality modules
│   ├── data_ingestion.py    # Data loading utilities
│   ├── data_splitter.py     # Train-test splitting
│   ├── feature_binning.py   # Feature discretization
│   ├── feature_encoding.py  # Categorical encoding
│   ├── feature_scaling.py   # Feature normalization
│   ├── handle_missing_values.py # Imputation strategies
│   ├── model_building.py    # Model architecture
│   ├── model_evaluation.py  # Performance metrics
│   ├── model_inference.py   # Prediction service
│   ├── model_training.py    # Training utilities
│   └── outlier_detection.py # Outlier handling
├── utils/                   # Helper utilities
│   └── config.py            # Configuration management
├── config.yaml              # Configuration parameters
├── Makefile                 # Automation commands
└── requirements.txt         # Dependencies
```

## ✨ Features

- 🧹 Comprehensive data preprocessing pipeline
  - Missing value imputation
  - Outlier detection and handling
  - Feature binning and encoding
  - Feature scaling

- 🧠 Flexible model training
  - Multiple algorithm support
  - Cross-validation
  - Hyperparameter tuning

- 📊 Robust model evaluation
  - Performance metrics calculation
  - Model comparison

- 🔄 Production-ready inference pipeline
  - Streaming prediction capability
  - Model versioning
  
- ⚙️ Configurable pipeline
  - YAML-based configuration
  - Easy parameter tuning

## 📋 Requirements

- Python 3.11+
- Pandas
- NumPy
- Scikit-learn
- Additional packages listed in `requirements.txt`

## 🛠️ Installation

1. Clone this repository:

```bash
git clone https://github.com/deaneeth/churn-prediction-production-pipeline.git
cd churn-prediction-production-pipeline
```

1. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

1. Install the required packages:

```bash
pip install -r requirements.txt
```

## 📝 Usage

### Data Preprocessing Pipeline

```python
from pipelines.data_pipeline import data_pipeline

# Run the complete data preprocessing pipeline
data = data_pipeline(data_path="data/raw/ChurnModelling.csv")
```

### Model Training Pipeline

```python
from pipelines.training_pipeline import train_model

# Train and evaluate the model
model, metrics = train_model(model_type="random_forest")
```

### Inference Pipeline

```python
from pipelines.streaming_inference_pipeline import predict

# Make predictions on new data
predictions = predict(input_data)
```

## 🔧 Pipeline Components

### Data Pipeline

The data pipeline handles:

- Data ingestion from CSV files
- Missing value imputation (mean, mode, custom strategies)
- Outlier detection using IQR or Z-score methods
- Feature binning for numeric variables
- Encoding of categorical variables
- Feature scaling
- Train-test splitting

### Training Pipeline

The training pipeline implements:

- Model selection from multiple algorithms
- Model training with cross-validation
- Hyperparameter tuning (optional)
- Performance evaluation
- Model persistence

### Inference Service

The inference pipeline provides:

- Loading of trained models
- Data preprocessing for new inputs
- Prediction generation
- Result formatting

## ⚙️ Configuration

All pipeline parameters are configured in `config.yaml`. Key configuration sections include:

- **Data Paths**: Locations of raw data, processed data, and artifacts
- **Columns**: Target variable, feature columns, columns to drop
- **Data Preprocessing**: Strategies for handling missing values, outliers, etc.
- **Feature Engineering**: Binning, encoding, scaling parameters
- **Training**: Model type, training strategy, hyperparameter tuning

## 📈 Model Performance

The pipeline includes robust evaluation metrics for model performance, including:

- Accuracy
- Precision, Recall, F1-score
- ROC AUC
- Confusion matrix

Performance metrics are calculated during model training and can be accessed through the training pipeline output.

## 🚀 Deployment

This project is designed to be deployed in a production environment. The inference pipeline supports streaming predictions for real-time applications.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.