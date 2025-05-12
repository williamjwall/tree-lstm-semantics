#!/bin/bash

echo "Starting setup process..."

# Upgrade pip
pip install --upgrade pip

# Install dependencies from requirements.txt
pip install -r requirements.txt

echo "Installing spaCy models..."
python -c "import spacy; spacy.cli.download('en_core_web_sm', '--user')" || echo "Will download models at runtime"

echo "Setup complete. Models will be downloaded at runtime if needed." 