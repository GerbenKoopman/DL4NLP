"""
Model Caching System
Singleton pattern for caching Gemma model to avoid reloading
"""

import logging
import threading
import time
from typing import Optional
from gemma import GemmaTranslationModel

logger = logging.getLogger(__name__)


class GemmaModelCache:
    """Singleton cache for Gemma translation model with TTL"""

    _instance = None
    _model: Optional[GemmaTranslationModel] = None
    _lock = threading.Lock()
    _cache_time: Optional[float] = None
    _model_name: Optional[str] = None

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def get_model(
        self,
        model_name: str = "google/gemma-3-270m-it",
        device: Optional[str] = None,
        token: Optional[str] = None,
        ttl_seconds: Optional[int] = 3600,  # Default TTL: 1 hour
    ) -> GemmaTranslationModel:
        """Get cached model or create a new one, with TTL and model name check."""
        with self._lock:
            now = time.time()
            is_stale = (
                self._cache_time is not None
                and ttl_seconds is not None
                and (now - self._cache_time > ttl_seconds)
            )
            model_changed = self._model_name != model_name

            if self._model is None or is_stale or model_changed:
                if is_stale:
                    logger.info("Cache expired (TTL). Reloading model.")
                elif model_changed and self._model is not None:
                    logger.info(
                        f"Model changed from '{self._model_name}' to '{model_name}'. Reloading."
                    )

                logger.info(f"🔄 Loading '{model_name}' into cache...")
                self._model = GemmaTranslationModel(
                    model_name, device=device, token=token
                )
                if not self._model.load_model():
                    self._model = None  # Ensure model is None on failure
                    raise RuntimeError(f"Failed to load Gemma model: {model_name}")

                self._cache_time = now
                self._model_name = model_name
                logger.info("✅ Gemma model cached successfully!")

            return self._model

    def clear_cache(self):
        """Clear the cached model and its metadata."""
        with self._lock:
            if self._model is not None:
                logger.info("🗑️ Clearing model cache...")
                self._model = None
                self._cache_time = None
                self._model_name = None

    def is_cached(self) -> bool:
        """Check if a model is currently cached."""
        return self._model is not None


# Convenience function for easy access
def get_cached_gemma_model(
    model_name: str = "google/gemma-3-270m-it",
    device: Optional[str] = None,
    token: Optional[str] = None,
    ttl_seconds: Optional[int] = 3600,
) -> GemmaTranslationModel:
    """Get a cached Gemma model instance."""
    cache = GemmaModelCache()
    return cache.get_model(model_name, device, token, ttl_seconds)
