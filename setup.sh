#!/bin/bash

echo "Setting up Tree-LSTM visualization environment..."

# Download spaCy model using Python
python -c "import spacy; spacy.cli.download('en_core_web_sm')"

# Download benepar model
python -c "import benepar; benepar.download('benepar_en3')"

echo "Setup complete! Models have been downloaded." 