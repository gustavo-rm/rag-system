from abc import ABC, abstractmethod
from typing import List
from sentence_transformers.readers import InputExample

class TripletGenerator(ABC):
    """
    Classe base abstrata para geradores de dados de treinamento (tripletos).
    Padrão de arquitetura: Strategy Pattern.
    """
    @abstractmethod
    def generate(self, **kwargs) -> List[InputExample]:
        pass