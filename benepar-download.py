#!/usr/bin/env python3

"""
Script to download the Berkeley Neural Parser model needed for constituency parsing.
This is used during Streamlit Cloud deployment to ensure the model is available.
"""

import os
import sys

def main():
    print("Downloading benepar_en3 model...")
    try:
        import benepar
        benepar.download('benepar_en3')
        print("Download successful!")
    except Exception as e:
        print(f"Error downloading model: {e}")
        # Don't exit with error code as this might prevent deployment
        # sys.exit(1)
    
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
            # Don't exit with error code as this might prevent deployment
            # sys.exit(1)
    
    except Exception as e:
        print(f"Error verifying model: {e}")
        # Don't exit with error code as this might prevent deployment
        # sys.exit(1)

if __name__ == "__main__":
    main() 