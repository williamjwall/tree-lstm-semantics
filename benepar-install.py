#!/usr/bin/env python3

"""
Script to download and set up the Berkeley Neural Parser model needed for constituency parsing.
This script is designed to work with modern Python versions (3.8+) and new benepar/nltk versions.
"""

import os
import sys
import subprocess
import importlib.util
import shutil
import traceback

# Constants
NLTK_DATA_DIR = os.path.expanduser('~/nltk_data')
BENEPAR_MODEL = "benepar_en3"

def check_package_installed(package_name):
    """Check if a Python package is installed"""
    return importlib.util.find_spec(package_name) is not None

def pip_install(package):
    """Install a package using pip"""
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    print("Benepar Installation and Setup Script")
    print("=====================================")
    
    # Create NLTK data directory if it doesn't exist
    os.makedirs(NLTK_DATA_DIR, exist_ok=True)
    print(f"NLTK data directory: {NLTK_DATA_DIR}")
    
    # Set NLTK_DATA environment variable
    os.environ['NLTK_DATA'] = NLTK_DATA_DIR
    print(f"Set NLTK_DATA environment variable to: {NLTK_DATA_DIR}")
    
    # Check for required packages
    requirements = ["nltk", "benepar", "spacy"]
    for req in requirements:
        if not check_package_installed(req):
            print(f"{req} not found, installing...")
            pip_install(req)
    
    # Install benepar model via nltk
    try:
        import nltk
        print(f"NLTK version: {nltk.__version__}")
        
        # Add NLTK_DATA to search path
        nltk.data.path.append(NLTK_DATA_DIR)
        print("Added NLTK_DATA to NLTK's data path")
        
        # Download benepar model
        print(f"Attempting to download {BENEPAR_MODEL}...")
        nltk.download(BENEPAR_MODEL)
        print(f"{BENEPAR_MODEL} download complete")
    except Exception as e:
        print(f"Error with NLTK: {str(e)}")
        print(traceback.format_exc())
    
    # Verify benepar installation
    try:
        import benepar
        print(f"Benepar installed: {benepar.__file__}")
        
        # Try to download via benepar's method
        try:
            benepar.download(BENEPAR_MODEL)
            print("Benepar model download via benepar.download() complete")
        except Exception as e:
            print(f"Benepar download method failed: {str(e)}")
            print("This may be OK if the model was already downloaded via NLTK")
    
    except Exception as e:
        print(f"Error with Benepar: {str(e)}")
        print(traceback.format_exc())
    
    # Set up spaCy
    try:
        import spacy
        print(f"spaCy version: {spacy.__version__}")
        
        # Download spaCy model if needed
        try:
            nlp = spacy.load("en_core_web_sm")
            print("spaCy model loaded successfully")
            
            # Try to add benepar to the pipeline
            try:
                import benepar
                if "benepar" not in nlp.pipe_names:
                    nlp.add_pipe("benepar", config={"model": BENEPAR_MODEL})
                    print("Added benepar to spaCy pipeline successfully")
                    
                    # Test with a simple sentence
                    doc = nlp("This is a test sentence.")
                    sent = list(doc.sents)[0]
                    print(f"Test parse: {sent._.parse_string}")
                else:
                    print("Benepar already in spaCy pipeline")
            except Exception as e:
                print(f"Could not add benepar to pipeline: {str(e)}")
                print(traceback.format_exc())
        except Exception as e:
            print(f"Error loading spaCy model: {str(e)}")
            print("You may need to install it with: python -m spacy download en_core_web_sm")
    except Exception as e:
        print(f"Error with spaCy: {str(e)}")
        print(traceback.format_exc())
    
    print("\nSetup complete. Check the messages above for any errors.")
    print("If there were errors, you may need to install specific versions of packages.")
    print("Remember to update your requirements.txt file if you make any changes.")
    
if __name__ == "__main__":
    main() 