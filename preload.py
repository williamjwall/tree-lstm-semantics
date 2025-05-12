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
import traceback

def main():
    print("Running preload script for Tree-LSTM Visualizer")
    
    # Check numpy version
    print("Checking numpy version...")
    try:
        import numpy as np
        print(f"Numpy version: {np.__version__}")
        print(f"Numpy path: {np.__file__}")
    except Exception as e:
        print(f"Error with numpy: {e}")
    
    # Check if spaCy model is available
    print("Checking spaCy model...")
    try:
        import spacy
        if spacy.util.is_package("en_core_web_sm"):
            print("spaCy model is available")
        else:
            print("Warning: en_core_web_sm not found but should be installed via requirements.txt")
    except Exception as e:
        print(f"Error with spaCy model: {e}")
    
    # Ensure NLTK is installed
    print("Checking NLTK...")
    try:
        import nltk
        print(f"NLTK version: {nltk.__version__}")
        
        # Create NLTK data directory if it doesn't exist
        nltk_data_dir = os.path.expanduser('~/nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        print(f"NLTK data directory: {nltk_data_dir}")
        
        # Set NLTK_DATA environment variable
        os.environ['NLTK_DATA'] = nltk_data_dir
        print(f"Set NLTK_DATA environment variable to: {nltk_data_dir}")
    except Exception as e:
        print(f"Error with NLTK: {e}")
    
    # Ensure benepar model is downloaded
    print("Checking benepar model...")
    try:
        import benepar
        
        # Try to get benepar version
        try:
            version = benepar.__version__
            print(f"Benepar version: {version}")
        except:
            print("Benepar version not available")
        
        # Check if model exists
        model_exists = False
        try:
            import benepar.download as benepar_download
            for path in benepar_download._get_download_dir():
                model_path = os.path.join(path, "benepar_en3")
                if os.path.exists(model_path):
                    model_exists = True
                    print(f"Found benepar_en3 model at: {model_path}")
                    break
        except Exception as e:
            print(f"Error checking benepar model path: {e}")
        
        if not model_exists:
            print("Downloading benepar model...")
            try:
                benepar.download('benepar_en3')
                print("Benepar model download completed")
                
                # Verify download
                for path in benepar_download._get_download_dir():
                    model_path = os.path.join(path, "benepar_en3")
                    if os.path.exists(model_path):
                        print(f"Verified benepar_en3 model at: {model_path}")
                        model_exists = True
                        break
                
                if not model_exists:
                    print("WARNING: Benepar model download claimed success but model not found")
            except Exception as e:
                print(f"Error downloading benepar model: {e}")
                print(traceback.format_exc())
        else:
            print("benepar model already downloaded")
    except Exception as e:
        print(f"Error with benepar: {e}")
        print(traceback.format_exc())
    
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