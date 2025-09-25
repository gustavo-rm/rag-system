from typing import List
from .base import QueryTransformer

class NoOpTransformer(QueryTransformer):
    """Uma implementação que não faz nada. Retorna a consulta original."""
    def transform(self, query: str) -> List[str]:
        print("Usando estratégia: Nenhuma Transformação (No-Op).")
        return [query]