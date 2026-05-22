import os
import sys
import nltk
import importlib
import subprocess
from typing import List, Optional, Tuple
import traceback

from src.tree_lstm_viz.logger import app_logger, log_dependency_status, log_model_status

class BeneparHelper:
    """Helper class for Benepar installation and model downloading."""
    
    def __init__(self, model_name: str = 'benepar_en3'):
        self.model_name = model_name
        self.nltk_data_dir = os.path.expanduser('~/nltk_data')
        os.makedirs(self.nltk_data_dir, exist_ok=True)
        os.environ['NLTK_DATA'] = self.nltk_data_dir
    
    def ensure_benepar_installed(self) -> bool:
        """
        Check if Benepar is installed, and attempt to install if not.
        
        Returns:
            bool: Whether Benepar is successfully installed
        """
        try:
            import benepar
            version = getattr(benepar, '__version__', None)
            log_dependency_status("Benepar", version=version, status="installed")
            return True
        except ImportError:
            log_dependency_status("Benepar", status="not installed")
            
            try:
                app_logger.info("Installing Benepar via pip...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "benepar"])
                
                # Try importing again
                import benepar
                version = getattr(benepar, '__version__', None)
                log_dependency_status("Benepar", version=version, status="installed successfully")
                return True
            except Exception as e:
                log_dependency_status("Benepar", status="installation failed", error=e)
                return False
    
    def get_model_paths(self) -> List[str]:
        """Get all possible paths where Benepar models could be stored."""
        paths = []
        
        # Try NLTK data path
        paths.extend(nltk.data.path)
        
        # Try benepar's paths if available
        try:
            import benepar.download as benepar_download
            if hasattr(benepar_download, '_get_download_dir'):
                paths.extend(benepar_download._get_download_dir())
        except ImportError:
            app_logger.debug("Couldn't import benepar.download")
        
        # Add standard Benepar model location
        paths.append(os.path.join(self.nltk_data_dir, 'models'))
        
        return paths
    
    def is_model_downloaded(self) -> Tuple[bool, Optional[str]]:
        """
        Check if the Benepar model is already downloaded.
        
        Returns:
            Tuple[bool, Optional[str]]: (is_downloaded, model_path)
        """
        for path in self.get_model_paths():
            model_path = os.path.join(path, self.model_name)
            if os.path.exists(model_path):
                log_model_status(self.model_name, path=model_path, status="found")
                return True, model_path
        
        log_model_status(self.model_name, status="not found")
        return False, None
    
    def download_model(self) -> bool:
        """
        Download the Benepar model using multiple fallback methods.
        
        Returns:
            bool: Whether the download was successful
        """
        # Check if already downloaded
        is_downloaded, path = self.is_model_downloaded()
        if is_downloaded:
            return True
        
        # Method 1: Use benepar's download method
        try:
            app_logger.info(f"Attempting to download {self.model_name} using benepar.download...")
            import benepar
            benepar.download(self.model_name)
            
            # Verify download
            is_downloaded, path = self.is_model_downloaded()
            if is_downloaded:
                log_model_status(self.model_name, path=path, status="downloaded via benepar.download")
                return True
        except Exception as e:
            log_model_status(self.model_name, status="download via benepar.download failed", error=e)
        
        # Method 2: Use NLTK's download
        try:
            app_logger.info(f"Attempting to download {self.model_name} using nltk.download...")
            nltk.download(self.model_name)
            
            # Verify download
            is_downloaded, path = self.is_model_downloaded()
            if is_downloaded:
                log_model_status(self.model_name, path=path, status="downloaded via nltk.download")
                return True
        except Exception as e:
            log_model_status(self.model_name, status="download via nltk.download failed", error=e)
        
        # Method 3: Try direct download via Python script
        try:
            app_logger.info("Attempting direct download method...")
            current_dir = os.path.dirname(os.path.abspath(__file__))
            downloader_path = os.path.join(current_dir, '..', '..', 'benepar-download.py')
            
            if os.path.exists(downloader_path):
                subprocess.check_call([sys.executable, downloader_path])
                
                # Verify download
                is_downloaded, path = self.is_model_downloaded()
                if is_downloaded:
                    log_model_status(self.model_name, path=path, status="downloaded via direct download script")
                    return True
        except Exception as e:
            log_model_status(self.model_name, status="download via direct script failed", error=e)
        
        app_logger.error(f"Failed to download {self.model_name} using all available methods")
        return False
    
    def setup_spacy_pipeline(self, nlp):
        """
        Add Benepar to a spaCy pipeline with robust error handling.
        
        Args:
            nlp: A spaCy language model
        
        Returns:
            The modified nlp pipeline
        
        Raises:
            RuntimeError: If Benepar cannot be added or constituency parsing fails
        """
        # CPU-only hosts (e.g. Streamlit Cloud): CUDA wheels + recent transformers can
        # leave Benepar's T5 submodule on PyTorch "meta" device and fail at first parse.
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

        import torch
        from transformers import AutoConfig, AutoModel

        _original_from_config = AutoModel.from_config

        def _from_config_materialized(config, *args, **kwargs):
            model = _original_from_config(config, *args, **kwargs)
            try:
                return model.to(torch.device("cpu"))
            except NotImplementedError:
                model_path = getattr(config, "_name_or_path", None) or getattr(
                    config, "name_or_path", None
                )
                if model_path and os.path.isdir(str(model_path)):
                    return AutoModel.from_pretrained(
                        model_path,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=False,
                    )
                raise

        AutoModel.from_config = _from_config_materialized

        # Import registers the spaCy "benepar" factory
        import benepar  # noqa: F401

        if not self.ensure_benepar_installed():
            raise RuntimeError("Benepar is not installed")

        if not self.download_model():
            raise RuntimeError(
                f"Benepar model '{self.model_name}' is not available. "
                "Run: python -c \"import benepar; benepar.download('benepar_en3')\""
            )

        if 'benepar' in nlp.pipe_names:
            app_logger.info("Benepar already in spaCy pipeline")
            return nlp

        try:
            nlp.add_pipe('benepar', config={'model': self.model_name})
            app_logger.info("Added benepar to spaCy pipeline")

            # Keep parser on CPU even if a CUDA build of torch is installed
            benepar_pipe = nlp.get_pipe("benepar")
            parser = getattr(benepar_pipe, "_parser", None)
            if parser is not None:
                parser.cpu()
                parser.eval()

            # Verify constituency parse is available
            doc = nlp("This is a test sentence.")
            sent = list(doc.sents)[0]
            _ = list(sent._.children)
            app_logger.info("Benepar parse tree generation working correctly")
        except Exception as e:
            app_logger.error(f"Failed to add Benepar to spaCy pipeline: {str(e)}")
            app_logger.debug(traceback.format_exc())
            raise RuntimeError(f"Failed to set up Benepar: {e}") from e

        return nlp 