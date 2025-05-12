#!/usr/bin/env python3
"""
Script to download required NLP models for the Tree-LSTM visualizer.
This is intended to be run during deployment.
"""

import os
import sys
import time

def main():
    print("Downloading required models...")
    
    # Download spaCy model
    print("Downloading spaCy model...")
    try:
        import spacy
        spacy.cli.download("en_core_web_sm")
        print("SpaCy model download completed")
    except Exception as e:
        print(f"Error downloading spaCy model: {e}")
        sys.exit(1)
    
    # Verify spaCy model
    try:
        if spacy.util.is_package("en_core_web_sm"):
            print("SpaCy model verified successfully")
        else:
            print("SpaCy model not found after download")
            sys.exit(1)
    except Exception as e:
        print(f"Error verifying spaCy model: {e}")
        sys.exit(1)
    
    # Download benepar model
    print("Downloading Berkeley Neural Parser model...")
    try:
        import benepar
        benepar.download('benepar_en3')
        print("Berkeley Neural Parser model download completed")
    except Exception as e:
        print(f"Error downloading benepar model: {e}")
        sys.exit(1)
    
    print("All models downloaded successfully!")

if __name__ == "__main__":
    main() 