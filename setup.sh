#!/bin/bash

set -e  # Exit on error

echo "Setting up Tree-LSTM visualization environment..."

# Check Python environment
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found"
    exit 1
fi

# Check if python3-venv is installed
if ! dpkg -l | grep -q python3-venv; then
    echo "Installing python3-venv..."
    sudo apt-get update
    sudo apt-get install -y python3-venv
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Create models directory if it doesn't exist
mkdir -p models

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Download spaCy model
echo "Downloading spaCy model..."
python -m spacy download en_core_web_sm

# Download Benepar model (using Python code rather than CLI)
echo "Downloading Benepar model..."
python -c "import benepar; benepar.download('benepar_en3')"

# Download BERT model
echo "Pre-downloading BERT model..."
python -c "
from transformers import BertModel, BertTokenizer
import torch

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Download models
print('Downloading BERT model...')
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Move model to appropriate device
model = model.to(device)
print('BERT model downloaded successfully')
"

# Deactivate virtual environment
deactivate

echo "Setup complete! All models have been downloaded."
echo "To run the application:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run the app: streamlit run streamlit_app.py" 