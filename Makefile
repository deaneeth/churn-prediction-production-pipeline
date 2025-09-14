.PHONY: all clean install train-pipeline data-pipeline streaming-inference run-all help

# Default Python interpreter
PYTHON = python
VENV = .venv/bin/activate

# Default target
all: help

# Help target
help:
	@echo "Available targets:"
	@echo "  make install             - Install project dependencies and set up environment"
	@echo "  make data-pipeline       - Run the data pipeline"
	@echo "  make train-pipeline      - Run the training pipeline"
	@echo "  make streaming-inference - Run the streaming inference pipeline with the sample JSON"
	@echo "  make run-all             - Run all pipelines in sequence"
	@echo "  make clean               - Clean up artifacts"

# Install project dependencies and set up environment
install:
	@echo "Installing project dependencies and setting up environment..."
	@echo "Creating virtual environment..."
	@python -m venv .venv
	@echo "Activating virtual environment and installing dependencies..."
	@.venv\Scripts\pip install --upgrade pip
	@.venv\Scripts\pip install -r requirements.txt
	@echo "Installation completed successfully!"
	@echo "To activate the virtual environment, run: .venv\Scripts\activate"

# Clean up
clean:
	@echo "Cleaning up artifacts..."
	rm -rf artifacts/models/*
	rm -rf artifacts/evaluation/*
	rm -rf artifacts/predictions/*
	rm -rf data/processed/*
	@echo "Cleanup completed!"



# Run data pipeline
data-pipeline:
	@echo "Running data pipeline..."
	@.venv\Scripts\python pipelines/data_pipeline.py

# Run training pipeline
train-pipeline:
	@echo "Running training pipeline..."
	@.venv\Scripts\python pipelines/training_pipeline.py

# Run streaming inference pipeline with sample JSON
streaming-inference:
	@echo "Running streaming inference pipeline with sample JSON..."
	@.venv\Scripts\python pipelines/streaming_inference_pipeline.py

# Run all pipelines in sequence
run-all:
	@echo "Running all pipelines in sequence..."
	@echo "========================================"
	@echo "Step 1: Running data pipeline"
	@echo "========================================"
	@.venv\Scripts\python pipelines/data_pipeline.py
	@echo "========================================"
	@echo "Step 2: Running training pipeline"
	@echo "========================================"
	@.venv\Scripts\python pipelines/training_pipeline.py
	@echo "========================================"
	@echo "Step 3: Running streaming inference pipeline"
	@echo "========================================"
	@.venv\Scripts\python pipelines/streaming_inference_pipeline.py
	@echo "========================================"
	@echo "All pipelines completed successfully!"
	@echo "========================================"