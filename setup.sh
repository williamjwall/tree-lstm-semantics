#!/bin/bash

# This script runs during Streamlit Cloud deployment to set up the environment

echo "Running setup.sh"

# Update pip
pip install --upgrade pip

# Clean any cached packages
pip cache purge

# Force reinstall numpy first to ensure binary compatibility
pip install --force-reinstall --no-cache-dir numpy==1.24.3

# Install scikit-learn separately to ensure compatibility with numpy
pip install --force-reinstall --no-cache-dir scikit-learn==1.2.2

# Install other requirements
pip install --no-cache-dir -r requirements.txt

# Verify numpy installation
python -c "import numpy as np; print(f'Numpy version: {np.__version__}, path: {np.__file__}')"

# Verify scikit-learn installation
python -c "import sklearn; print(f'Scikit-learn version: {sklearn.__version__}, path: {sklearn.__file__}')"

# Run preload script to download models
python preload.py

echo "Setup complete!" 