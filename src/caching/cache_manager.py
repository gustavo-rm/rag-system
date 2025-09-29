from typing import Optional, Dict

class CacheManager:
    """
    Gerencia um cache de chave-valor simples para respostas exatas.
    Esta implementação usa um dicionário em memória, mas pode ser facilmente
    estendida para usar um backend como Redis para persistência e compartilhamento.
    """
    def __init__(self):
        self._cache: Dict[str, str] = {}
        print("Cache Manager (Exato, Camada 1) inicializado.")

    def _normalize_key(self, key: str) -> str:
        """Normaliza a chave para consistência no cache."""
        return key.strip().lower()

    def get(self, key: str) -> Optional[str]:
        """Busca um valor no cache usando uma chave."""
        normalized_key = self._normalize_key(key)
        return self._cache.get(normalized_key)

    def set(self, key: str, value: str):
        """Define um valor no cache para uma determinada chave."""
        normalized_key = self._normalize_key(key)
        self._cache[normalized_key] = value