from abc import ABC, abstractmethod
from typing import List
from sentence_transformers.readers import InputExample

class TripletGenerator(ABC):
    """
    Abstract base class for training data generators (triplets).
    Architecture Pattern: Strategy Pattern.
    """
    @abstractmethod
    def generate(self, **kwargs) -> List[InputExample]:
        """
        Generates a list of InputExample objects for training.
        Each InputExample typically contains [anchor, positive, negative].
        """
        pass
