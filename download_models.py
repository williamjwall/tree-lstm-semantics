#!/usr/bin/env python3
"""
Model downloader for Tree-LSTM Visualizer
This script ensures all required models are downloaded before the app starts
"""
import sys
import os
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("model_downloader")

def ensure_spacy_model():
    """Ensure the spaCy model is downloaded and available."""
    try:
        logger.info("Checking for spaCy model...")
        import spacy
        try:
            # Try loading the model
            nlp = spacy.load("en_core_web_sm")
            logger.info("✓ spaCy model already available")
            return True
        except OSError:
            logger.info("spaCy model not found, downloading...")
            
            # Try direct download method
            try:
                spacy.cli.download("en_core_web_sm")
                logger.info("✓ spaCy model downloaded successfully")
                return True
            except Exception as e:
                logger.warning(f"spaCy download via API failed: {e}")
            
            # Try subprocess method
            try:
                subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
                logger.info("✓ spaCy model downloaded successfully via subprocess")
                return True
            except subprocess.SubprocessError as e:
                logger.error(f"Failed to download spaCy model: {e}")
                return False
    except ImportError:
        logger.error("spaCy package not installed")
        return False

def ensure_benepar_model():
    """Ensure the Berkeley Neural Parser model is downloaded and available."""
    try:
        logger.info("Checking for benepar model...")
        import benepar
        
        # Check if model directory exists
        home_dir = Path.home()
        benepar_dir = home_dir / ".benepar_downloads"
        model_path = benepar_dir / "benepar_en3"
        
        if model_path.exists():
            logger.info("✓ benepar model already available")
            return True
        else:
            logger.info("benepar model not found, downloading...")
            try:
                benepar.download("benepar_en3")
                logger.info("✓ benepar model downloaded successfully")
                return True
            except Exception as e:
                logger.error(f"Failed to download benepar model: {e}")
                return False
    except ImportError:
        logger.error("benepar package not installed")
        return False

def main():
    """Main function to ensure all models are downloaded."""
    logger.info("Starting model download check...")
    
    spacy_status = ensure_spacy_model()
    benepar_status = ensure_benepar_model()
    
    if spacy_status and benepar_status:
        logger.info("All models successfully downloaded and verified!")
        return 0
    else:
        missing = []
        if not spacy_status:
            missing.append("spaCy")
        if not benepar_status:
            missing.append("benepar")
        
        logger.error(f"Failed to download the following models: {', '.join(missing)}")
        logger.info("The app may still function with limited capabilities")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 