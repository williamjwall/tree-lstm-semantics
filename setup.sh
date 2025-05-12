#!/bin/bash

# This script runs during Streamlit Cloud deployment to set up the environment

echo "Running setup.sh"

# Update pip
pip install --upgrade pip

# Clean any cached packages and remove previous installations
pip cache purge
pip uninstall -y numpy scikit-learn spacy

# Install requirements with exact versions to ensure compatibility
pip install --no-cache-dir numpy==1.26.3
pip install --no-cache-dir scikit-learn==1.3.2
pip install --no-cache-dir -r requirements.txt

# Verify numpy installation
python -c "import numpy as np; print(f'Numpy version: {np.__version__}, path: {np.__file__}')"

# Verify scikit-learn installation
python -c "import sklearn; print(f'Scikit-learn version: {sklearn.__version__}, path: {sklearn.__file__}')"

# Run preload script to download models
python preload.py

echo "Setup complete!" 