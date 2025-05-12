#!/bin/bash

# This script runs during Streamlit Cloud deployment to set up the environment

echo "Running setup.sh"

# Update pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Run preload script to download models
python preload.py

echo "Setup complete!" 