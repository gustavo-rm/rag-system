from abc import ABC, abstractmethod
from typing import List

class QueryTransformer(ABC):
    @abstractmethod
    def transform(self, query: str) -> List[str]:
        """
        Transforma uma única consulta de entrada em uma ou mais consultas otimizadas para busca.
        Retorna sempre uma lista de strings.
        """
        pass