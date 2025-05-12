#!/usr/bin/env python3

"""
Script to download the Berkeley Neural Parser model needed for constituency parsing.
This is used during Streamlit Cloud deployment to ensure the model is available.
"""

import os
import sys
import traceback

def main():
    print("Downloading benepar_en3 model...")
    
    # Create NLTK data directory if it doesn't exist
    nltk_data_dir = os.path.expanduser('~/nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    print(f"NLTK data directory: {nltk_data_dir}")
    
    # Set NLTK_DATA environment variable
    os.environ['NLTK_DATA'] = nltk_data_dir
    print(f"Set NLTK_DATA environment variable to: {nltk_data_dir}")
    
    # Install NLTK if needed
    try:
        import nltk
        print(f"NLTK version: {nltk.__version__}")
    except ImportError:
        print("NLTK not found, installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])
        import nltk
        print(f"Installed NLTK version: {nltk.__version__}")
    
    try:
        import benepar
        benepar.download('benepar_en3')
        print("Download successful!")
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        print(traceback.format_exc())
    
    # Verify download
    try:
        import benepar.download as benepar_download
        download_dir = benepar_download._get_download_dir()
        
        found = False
        for path in download_dir:
            model_path = os.path.join(path, "benepar_en3")
            if os.path.exists(model_path):
                print(f"Model verified at: {model_path}")
                found = True
                break
        
        if not found:
            print("Warning: Model was downloaded but could not be verified in the paths:")
            for path in download_dir:
                print(f"  - {path}")
            
            # Try alternative download method
            print("Trying alternative download method...")
            try:
                # Use NLTK downloader directly
                nltk.download('benepar_en3', download_dir=nltk_data_dir)
                print("NLTK download completed")
                
                # Check again
                found = False
                for path in download_dir:
                    model_path = os.path.join(path, "benepar_en3")
                    if os.path.exists(model_path):
                        print(f"Model verified at: {model_path}")
                        found = True
                        break
                
                if not found:
                    print("Warning: Model still not found after alternative download")
            except Exception as e:
                print(f"Error with alternative download: {str(e)}")
    
    except Exception as e:
        print(f"Error verifying model: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 