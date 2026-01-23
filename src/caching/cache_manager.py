import logging
from typing import Optional, Dict

# Logger Configuration
logger = logging.getLogger(__name__)


class CacheManager:
    """
    Manages an in-memory exact match cache (Key-Value).

    Ideal for quickly capturing identical questions before triggering heavy
    vector calculations. Implements a simple capacity policy to avoid memory leaks.
    """

    def __init__(self, capacity: int = 10000):
        """
        Initializes the cache manager.

        Args:
            capacity (int): Maximum number of items to store.
                            When full, the current behavior is to clear (flush)
                            for simplicity, or it could be LRU in the future.
        """
        self._cache: Dict[str, str] = {}
        self.capacity = capacity
        logger.info(f"💾 Cache Manager (Layer 1 - Exact) initialized. Capacity: {capacity}")

    def _normalize_key(self, key: str) -> str:
        """
        Normalizes the key to ensure hits even with minor formatting variations.

        Applies: strip (remove leading/trailing spaces) and lower (lowercase).

        Args:
            key (str): The original key to normalize.

        Returns:
            str: The normalized key.
        """
        return key.strip().lower()

    def get(self, key: str) -> Optional[str]:
        """
        Retrieves a value from the cache.

        Args:
            key (str): The user's question.

        Returns:
            Optional[str]: The stored response or None if there is no hit.
        """
        normalized_key = self._normalize_key(key)
        value = self._cache.get(normalized_key)

        if value:
            logger.info(f"🎯 Exact Cache HIT for: '{key[:30]}...'")
        else:
            logger.debug(f"Exact Cache MISS for: '{key[:30]}...'")

        return value

    def set(self, key: str, value: str):
        """
        Stores a question/answer pair.

        Args:
            key (str): The original question.
            value (str): The generated response.
        """
        # Basic memory protection
        if len(self._cache) >= self.capacity:
            logger.warning("Exact Cache reached maximum capacity. Clearing memory (Flush).")
            self._cache.clear()

        normalized_key = self._normalize_key(key)
        self._cache[normalized_key] = value
        logger.debug(f"Saved to Exact Cache: '{key[:30]}...'")
