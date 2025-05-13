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

# Add src directory to path to allow importing our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    # First try to use our new structured logging
    try:
        from src.tree_lstm_viz.logger import setup_logger, log_dependency_status, log_model_status
        logger = setup_logger('preload')
        logger.info("Running preload script for Tree-LSTM Visualizer")
        structured_logging = True
    except ImportError:
        print("Running preload script for Tree-LSTM Visualizer")
        structured_logging = False
    
    # Check numpy version
    try:
        import numpy as np
        version = np.__version__
        if structured_logging:
            log_dependency_status("Numpy", version=version, status="loaded")
        else:
            print(f"Numpy version: {version}")
    except Exception as e:
        if structured_logging:
            log_dependency_status("Numpy", status="failed to load", error=e)
        else:
            print(f"Error with numpy: {e}")
    
    # Check scikit-learn version
    try:
        import sklearn
        version = sklearn.__version__
        if structured_logging:
            log_dependency_status("Scikit-learn", version=version, status="loaded")
        else:
            print(f"Scikit-learn version: {version}")
    except Exception as e:
        if structured_logging:
            log_dependency_status("Scikit-learn", status="failed to load", error=e)
        else:
            print(f"Error with scikit-learn: {e}")
    
    # Check if spaCy model is available
    try:
        import spacy
        version = spacy.__version__
        if structured_logging:
            log_dependency_status("spaCy", version=version, status="loaded")
        else:
            print(f"spaCy version: {version}")
            
        if spacy.util.is_package("en_core_web_sm"):
            if structured_logging:
                log_model_status("en_core_web_sm", status="available")
            else:
                print("spaCy model is available")
        else:
            if structured_logging:
                log_model_status("en_core_web_sm", status="not found")
            else:
                print("Warning: en_core_web_sm not found but should be installed via requirements.txt")
    except Exception as e:
        if structured_logging:
            log_dependency_status("spaCy", status="failed to load", error=e)
        else:
            print(f"Error with spaCy model: {e}")
    
    # Ensure NLTK is installed
    try:
        import nltk
        version = nltk.__version__
        if structured_logging:
            log_dependency_status("NLTK", version=version, status="loaded")
        else:
            print(f"NLTK version: {version}")
        
        # Create NLTK data directory if it doesn't exist
        nltk_data_dir = os.path.expanduser('~/nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        
        # Set NLTK_DATA environment variable
        os.environ['NLTK_DATA'] = nltk_data_dir
        if structured_logging:
            logger.info(f"NLTK data directory set to: {nltk_data_dir}")
        else:
            print(f"Set NLTK_DATA environment variable to: {nltk_data_dir}")
    except Exception as e:
        if structured_logging:
            log_dependency_status("NLTK", status="failed to load", error=e)
        else:
            print(f"Error with NLTK: {e}")
    
    # Use our new BeneparHelper if available, otherwise fall back to manual checks
    try:
        from src.tree_lstm_viz.benepar_utils import BeneparHelper
        helper = BeneparHelper('benepar_en3')
        
        # Ensure benepar is installed
        helper.ensure_benepar_installed()
        
        # Download the model if needed
        helper.download_model()
    except ImportError:
        # Fall back to old implementation if BeneparHelper is not available
        if structured_logging:
            logger.warning("BeneparHelper not available, falling back to manual Benepar setup")
        else:
            print("BeneparHelper not available, falling back to manual Benepar setup")
            
        try:
            import benepar
            
            # Try to get benepar version
            try:
                version = benepar.__version__
                if structured_logging:
                    log_dependency_status("Benepar", version=version, status="loaded")
                else:
                    print(f"Benepar version: {version}")
            except:
                if structured_logging:
                    log_dependency_status("Benepar", status="loaded", version=None)
                else:
                    print("Benepar version not available")
            
            # Check if model exists
            model_exists = False
            try:
                import benepar.download as benepar_download
                for path in benepar_download._get_download_dir():
                    model_path = os.path.join(path, "benepar_en3")
                    if os.path.exists(model_path):
                        model_exists = True
                        if structured_logging:
                            log_model_status("benepar_en3", path=model_path, status="found")
                        else:
                            print(f"Found benepar_en3 model at: {model_path}")
                        break
            except Exception as e:
                if structured_logging:
                    log_model_status("benepar_en3", status="check failed", error=e)
                else:
                    print(f"Error checking benepar model path: {e}")
            
            if not model_exists:
                if structured_logging:
                    logger.info("Downloading benepar model...")
                else:
                    print("Downloading benepar model...")
                try:
                    benepar.download('benepar_en3')
                    if structured_logging:
                        log_model_status("benepar_en3", status="downloaded")
                    else:
                        print("Benepar model download completed")
                except Exception as e:
                    if structured_logging:
                        log_model_status("benepar_en3", status="download failed", error=e)
                    else:
                        print(f"Error downloading benepar model: {e}")
                        print(traceback.format_exc())
            else:
                if structured_logging:
                    logger.info("benepar model already downloaded")
                else:
                    print("benepar model already downloaded")
        except Exception as e:
            if structured_logging:
                log_dependency_status("Benepar", status="failed to load", error=e)
            else:
                print(f"Error with benepar: {e}")
                print(traceback.format_exc())
    
    # Ensure transformers models are downloaded
    try:
        from transformers import BertModel, BertTokenizer
        
        # Just import to trigger download
        if structured_logging:
            logger.info("Downloading BERT model and tokenizer...")
        else:
            print("Downloading BERT model and tokenizer...")
            
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        
        if structured_logging:
            log_model_status("BERT", status="downloaded")
        else:
            print("BERT model and tokenizer downloaded")
    except Exception as e:
        if structured_logging:
            log_model_status("BERT", status="download failed", error=e)
        else:
            print(f"Error with BERT model: {e}")
    
    if structured_logging:
        logger.info("Preload complete!")
    else:
        print("Preload complete!")

if __name__ == "__main__":
    main() 