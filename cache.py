"""
Model Caching System
Singleton pattern for caching Gemma model to avoid reloading
"""

import logging
import threading
from typing import Optional
from gemma import GemmaTranslationModel

logger = logging.getLogger(__name__)

class GemmaModelCache:
    """Singleton cache for Gemma translation model"""
    
    _instance = None
    _model = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_model(self, model_name: str = "google/gemma-3-270m-it", 
                  device: Optional[str] = None) -> GemmaTranslationModel:
        """Get cached model or create new one"""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    logger.info("🔄 Loading Gemma model into cache...")
                    self._model = GemmaTranslationModel(model_name, device)
                    success = self._model.load_model()
                    if not success:
                        raise RuntimeError("Failed to load Gemma model")
                    logger.info("✅ Gemma model cached successfully!")
        return self._model
    
    def clear_cache(self):
        """Clear the cached model"""
        with self._lock:
            if self._model is not None:
                logger.info("🗑️ Clearing model cache...")
                del self._model
                self._model = None
    
    def is_cached(self) -> bool:
        """Check if model is cached"""
        return self._model is not None

# Convenience function for easy access
def get_cached_gemma_model(model_name: str = "google/gemma-3-270m-it", 
                          device: Optional[str] = None) -> GemmaTranslationModel:
    """Get cached Gemma model instance"""
    cache = GemmaModelCache()
    return cache.get_model(model_name, device)
