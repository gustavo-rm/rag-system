from typing import List
from .base import QueryTransformer
import logging
# Configuração de Logger
logger = logging.getLogger(__name__)


class NoOpTransformer(QueryTransformer):
    """Uma implementação que não faz nada. Retorna a consulta original."""
    def transform(self, query: str) -> List[str]:
        logger.info("Usando estratégia: Nenhuma Transformação (No-Op).")
        return [query]