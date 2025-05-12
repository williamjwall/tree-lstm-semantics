#!/bin/bash

# This script runs during Streamlit Cloud deployment to set up the environment

echo "Running setup.sh"

# Update pip
pip install --upgrade pip

# Clean any cached packages and remove previous installations
pip cache purge
pip uninstall -y numpy scikit-learn spacy benepar

# Install requirements with exact versions to ensure compatibility
pip install --no-cache-dir numpy==1.26.3
pip install --no-cache-dir scikit-learn==1.3.2
pip install --no-cache-dir -r requirements.txt

# Install NLTK
pip install --no-cache-dir nltk

# Create NLTK data directories if they don't exist
mkdir -p /home/appuser/nltk_data

# Download benepar model
python -c "import benepar; benepar.download('benepar_en3')"

# Verify model download
python -c "
import os
import sys
nltk_paths = ['/home/appuser/nltk_data', '/usr/share/nltk_data', '/usr/local/share/nltk_data']
for path in nltk_paths:
    model_path = os.path.join(path, 'models', 'benepar_en3')
    if os.path.exists(model_path):
        print(f'Found benepar_en3 at: {model_path}')
        sys.exit(0)
print('WARNING: benepar_en3 model not found in expected locations')
"

# Verify numpy installation
python -c "import numpy as np; print(f'Numpy version: {np.__version__}, path: {np.__file__}')"

# Verify scikit-learn installation
python -c "import sklearn; print(f'Scikit-learn version: {sklearn.__version__}, path: {sklearn.__file__}')"

# Run preload script to download models
python preload.py

echo "Setup complete!" 