#!/bin/bash

echo "Starting Tree-LSTM visualization environment setup..."

# Make script more robust
set -e

# Make pip installation more robust
pip install --upgrade pip setuptools wheel

# Install core dependencies without using cache
pip install --no-cache-dir -r requirements.txt

echo "Installing spaCy model..."
# Try multiple approaches to ensure model gets installed
python -m spacy download en_core_web_sm || python -c "import spacy; spacy.cli.download('en_core_web_sm')"

echo "Installing benepar model..."
python -c "import benepar; benepar.download('benepar_en3')" || echo "Benepar download will be retried during app startup"

# Verify spaCy model installation
if python -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null; then
    echo "✓ spaCy model installed successfully"
else
    echo "⚠️ spaCy model installation might have failed. Will retry at runtime."
fi

echo "Setup complete. App will attempt to download missing models at runtime if needed." 