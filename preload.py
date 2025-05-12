#!/usr/bin/env python3
"""
Preload script for Tree-LSTM visualizer.
This script is run before the Streamlit app starts to ensure all necessary
dependencies are downloaded and available.
"""

import os
import sys
import subprocess
import time

def main():
    print("Running preload script for Tree-LSTM Visualizer")
    
    # Ensure spaCy model is downloaded
    print("Checking spaCy model...")
    try:
        import spacy
        if not spacy.util.is_package("en_core_web_sm"):
            print("Downloading spaCy model...")
            spacy.cli.download("en_core_web_sm")
            print("SpaCy model download completed")
        else:
            print("spaCy model already downloaded")
    except Exception as e:
        print(f"Error with spaCy model: {e}")
    
    # Ensure benepar model is downloaded
    print("Checking benepar model...")
    try:
        import benepar
        import benepar.download as benepar_download
        
        # Check if model exists
        model_exists = False
        for path in benepar_download._get_download_dir():
            if os.path.exists(os.path.join(path, "benepar_en3")):
                model_exists = True
                break
        
        if not model_exists:
            print("Downloading benepar model...")
            benepar.download('benepar_en3')
        else:
            print("benepar model already downloaded")
    except Exception as e:
        print(f"Error with benepar model: {e}")
    
    # Ensure transformers models are downloaded
    print("Checking BERT model...")
    try:
        from transformers import BertModel, BertTokenizer
        
        # Just import to trigger download
        print("Downloading BERT model and tokenizer...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        print("BERT model and tokenizer downloaded")
    except Exception as e:
        print(f"Error with BERT model: {e}")
    
    print("Preload complete!")

if __name__ == "__main__":
    main() 