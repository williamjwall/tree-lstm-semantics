import logging
import os
import sys
from typing import Optional, Dict, Any, Union
import traceback

# Configure the basic logging system
def setup_logger(name: str = 'tree_lstm_visualizer', 
                 level: int = logging.INFO, 
                 log_file: Optional[str] = None,
                 console_output: bool = True) -> logging.Logger:
    """
    Set up a logger with consistent formatting and optional file output
    
    Args:
        name: Name of the logger
        level: Logging level (default: INFO)
        log_file: Optional file path to write logs
        console_output: Whether to output logs to console
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers
    
    # Define formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if a log file was specified
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Create a default application logger
app_logger = setup_logger()

# Custom methods to handle dependency loading messages
def log_dependency_status(module_name: str, version: Optional[str] = None, status: str = "loaded", error: Optional[Exception] = None):
    """
    Log dependency loading status in a consistent format
    
    Args:
        module_name: Name of the module/dependency
        version: Version string if available
        status: Status message (loaded, failed, etc.)
        error: Exception if there was an error
    """
    version_str = f"version: {version}" if version else "version not available"
    if error:
        app_logger.warning(f"{module_name} {version_str} - {status}: {str(error)}")
        app_logger.debug(traceback.format_exc())
    else:
        app_logger.info(f"{module_name} {version_str} - {status}")

def log_model_status(model_name: str, path: Optional[str] = None, status: str = "loaded", error: Optional[Exception] = None):
    """
    Log model loading status in a consistent format
    
    Args:
        model_name: Name of the model
        path: Path to the model if available
        status: Status message (loaded, failed, etc.)
        error: Exception if there was an error
    """
    path_str = f"path: {path}" if path else "path not available"
    if error:
        app_logger.warning(f"Model {model_name} {path_str} - {status}: {str(error)}")
        app_logger.debug(traceback.format_exc())
    else:
        app_logger.info(f"Model {model_name} {path_str} - {status}") 