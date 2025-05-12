#!/bin/bash

echo "Setting up Tree-LSTM visualization environment..."

# Download benepar model
python -c "import benepar; benepar.download('benepar_en3')"

echo "Setup complete! Models have been downloaded." 