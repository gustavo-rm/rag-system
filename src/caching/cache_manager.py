import logging
from typing import Optional, Dict

# Configuração de Logger
logger = logging.getLogger(__name__)


class CacheManager:
    """
    Gerencia um cache de correspondência exata (Key-Value) em memória.

    Ideal para capturar perguntas idênticas rapidamente antes de acionar 
    cálculos vetoriais pesados. Implementa uma política simples de capacidade
    para evitar vazamento de memória.
    """

    def __init__(self, capacity: int = 10000):
        """
        Inicializa o gerenciador de cache.

        Args:
            capacity (int): Número máximo de itens a armazenar. 
                            Quando cheio, o comportamento atual é limpar (flush) 
                            para simplicidade, ou poderia ser LRU no futuro.
        """
        self._cache: Dict[str, str] = {}
        self.capacity = capacity
        logger.info(f"💾 Cache Manager (Camada 1 - Exato) inicializado. Capacidade: {capacity}")

    def _normalize_key(self, key: str) -> str:
        """
        Normaliza a chave para garantir hits mesmo com pequenas variações de formatação.

        Aplica: strip (remove espaços nas pontas) e lower (minúsculas).
        """
        return key.strip().lower()

    def get(self, key: str) -> Optional[str]:
        """
        Busca um valor no cache.

        Args:
            key (str): A pergunta do usuário.

        Returns:
            Optional[str]: A resposta armazenada ou None se não houver hit.
        """
        normalized_key = self._normalize_key(key)
        value = self._cache.get(normalized_key)

        if value:
            logger.info(f"🎯 Cache Exato HIT para: '{key[:30]}...'")
        else:
            logger.debug(f"Cache Exato MISS para: '{key[:30]}...'")

        return value

    def set(self, key: str, value: str):
        """
        Armazena um par pergunta/resposta.

        Args:
            key (str): A pergunta original.
            value (str): A resposta gerada.
        """
        # Proteção básica de memória
        if len(self._cache) >= self.capacity:
            logger.warning("Cache Exato atingiu capacidade máxima. Limpando memória (Flush).")
            self._cache.clear()

        normalized_key = self._normalize_key(key)
        self._cache[normalized_key] = value
        logger.debug(f"Salvo no Cache Exato: '{key[:30]}...'")