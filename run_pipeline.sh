#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Step 1: Sync files with 'uv' command
echo "Syncing files with group 'neuralnetwork'..."
uv sync --group=neuralnetwork

# Step 2: Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Step 3: Run script to download files
echo "Running 00_download_files.py..."
python scripts/00_download_files.py

# Step 4: Run script for data preparation
echo "Running 01_data_preparation.py..."
python scripts/01_data_preparation.py

# Step 5: Run script for data preparation
echo "Running 02_model_creation.py..."
python scripts/02_model_creation.py
 

echo "Pipeline completed successfully!"