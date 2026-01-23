from abc import ABC, abstractmethod
from typing import List


class QueryTransformer(ABC):
    """
    Abstract Base Class for query transformation strategies.

    Query transformers are used to expand or modify a user's original query
    to improve retrieval performance in a RAG system.
    """

    @abstractmethod
    def transform(self, query: str) -> List[str]:
        """
        Transforms a single input query into one or more search-optimized queries.

        Args:
            query (str): The original user query.

        Returns:
            List[str]: A list of transformed queries (strings).
        """
        pass
