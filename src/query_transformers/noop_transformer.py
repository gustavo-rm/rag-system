from typing import List
from .base import QueryTransformer
import logging
# Logger Configuration
logger = logging.getLogger(__name__)


class NoOpTransformer(QueryTransformer):
    """
    An implementation that does nothing. Returns the original query.
    Used when the query is simple and specific enough.
    """

    def transform(self, query: str) -> List[str]:
        """
        Returns the original query wrapped in a list.

        Args:
            query (str): The original user query.

        Returns:
            List[str]: A list containing only the original query.
        """
        logger.info("Using strategy: No Transformation (No-Op).")
        return [query]
