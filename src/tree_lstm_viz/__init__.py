"""
Tree-LSTM Visualizer module.
Provides utilities for constituency parsing and tree visualization.
"""

__version__ = "0.1.0"

# Import key components for easier access, but make them optional
try:
    from src.tree_lstm_viz.logger import (
        setup_logger, 
        app_logger, 
        log_dependency_status, 
        log_model_status
    )
except ImportError:
    # Define fallbacks if imports fail
    def setup_logger(*args, **kwargs):
        import logging
        return logging.getLogger('tree_lstm_viz')
    
    app_logger = setup_logger()
    
    def log_dependency_status(module_name, version=None, status="unknown", error=None):
        if error:
            app_logger.warning(f"{module_name} {status}: {error}")
        else:
            version_str = f"{version}" if version else "version unknown"
            app_logger.info(f"{module_name} {version_str} - {status}")
    
    def log_model_status(model_name, path=None, status="unknown", error=None):
        if error:
            app_logger.warning(f"Model {model_name} {status}: {error}")
        else:
            path_str = f"at {path}" if path else "path unknown"
            app_logger.info(f"Model {model_name} {path_str} - {status}")

# Try to import BeneparHelper, but don't fail if it's not available
try:
    from src.tree_lstm_viz.benepar_utils import BeneparHelper
except ImportError:
    # Define a minimal fallback
    class BeneparHelper:
        def __init__(self, model_name='benepar_en3'):
            self.model_name = model_name
            
        def ensure_benepar_installed(self):
            return False
            
        def is_model_downloaded(self):
            return False, None
            
        def download_model(self):
            return False
            
        def setup_spacy_pipeline(self, nlp):
            return nlp

# Don't try to import model classes directly in __init__ to avoid dependency issues
# These will be imported when needed 