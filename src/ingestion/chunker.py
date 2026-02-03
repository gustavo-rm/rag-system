import re
from typing import List, Optional
from src.config import Config


class Chunker:
    """
    Responsible for dividing long texts into smaller pieces (chunks) while preserving semantic cohesion.

    Implements a "Sliding Window" strategy over semantic units.
    The text is first recursively broken down into the largest possible unit (Paragraph -> Sentence -> Word)
    and then regrouped to fill the chunk size, maintaining a context overlap.
    """

    def __init__(self, chunk_size: int = Config.CHUNK_SIZE, chunk_overlap: int = Config.CHUNK_OVERLAP):
        """
        Initializes the Chunker.

        Args:
            chunk_size (int): The desired maximum number of characters per chunk.
            chunk_overlap (int): Number of characters that should repeat between adjacent chunks
                                 to maintain context.

        Raises:
            ValueError: If chunk_overlap is greater than or equal to chunk_size.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly smaller than chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Hierarchy of separators: Double paragraph -> Paragraph -> Sentence -> Punctuation -> Space -> Character
        self.separators = ["\n\n", "\n", ". ", "? ", "! ", ";", " ", ""]

    def chunk_text(self, text: str) -> List[str]:
        """
        Main method to process raw text and return optimized chunks.

        Performs pre-cleaning on the text to normalize spaces before segmenting.

        Args:
            text (str): The raw text to be processed.

        Returns:
            List[str]: List of text chunks ready for vectorization.
        """
        if not text:
            return []

        # 1. Basic sanitization: remove excess whitespace (tabs, multiple spaces)
        # This avoids creating "phantom chunks" or wasting tokens.
        text = re.sub(r'\s+', ' ', text).strip()

        # 2. Recursive Split: Breaks the text into units smaller than chunk_size
        splits = self._recursive_split(text, self.separators)

        # 3. Merge with Overlap: Regroups units into final chunks
        return self._merge_splits(splits)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively divides the text until all pieces are smaller than chunk_size.

        Args:
            text (str): The text or fragment to be analyzed.
            separators (List[str]): List of available separators for splitting attempts.

        Returns:
            List[str]: A flat list of "atomic units" (e.g., sentences or words)
                       where each item is guaranteed to be smaller or equal to chunk_size
                       (unless a single word exceeds the limit).
        """
        final_splits = []

        # If the text already fits in the chunk, we don't need to split further.
        if len(text) <= self.chunk_size:
            return [text]

        # If we ran out of separators, we are forced to return the text as is
        # (rare case of a giant word) or split by character if the empty separator is in the list.
        if not separators:
            return [text]

        separator = separators[0]
        next_separators = separators[1:]

        # Tries to split by the current separator
        if separator == "":
            # Base case: split by character
            splits = list(text)
        else:
            # Keep the separator in the split.
            # The regex `({sep})` with parentheses captures the separator.
            _splits = re.split(f"({re.escape(separator)})", text)
            splits = []

            # Regroups the separator with the previous text to maintain correct punctuation.
            # Ex: ["Hello", ".", " World"] -> ["Hello.", " World"]
            for i in range(1, len(_splits), 2):
                splits.append(_splits[i - 1] + _splits[i])
            # Adds the last piece if left over (no separator at the end)
            if len(_splits) % 2 != 0:
                splits.append(_splits[-1])

        # Filter empty strings
        splits = [s for s in splits if s]

        # Evaluate each resulting piece
        for s in splits:
            if len(s) > self.chunk_size:
                # If the piece is still too big, recurse with the NEXT separator
                final_splits.extend(self._recursive_split(s, next_separators))
            else:
                final_splits.append(s)

        return final_splits

    def _merge_splits(self, splits: List[str]) -> List[str]:
        """
        Combines small text units into larger chunks, managing overlap.

        Unlike simple string slicing, this method removes whole units
        from the start of the buffer to create the overlap, ensuring we don't cut words in half.

        Args:
            splits (List[str]): List of text units (e.g., sentences).

        Returns:
            List[str]: Final list of chunks.
        """
        final_chunks = []
        current_chunk: List[str] = []
        current_length = 0

        for split in splits:
            split_len = len(split)

            # Checks if adding the new split exceeds the maximum size
            if current_length + split_len > self.chunk_size:

                # 1. Save the current chunk if it is not empty
                if current_length > 0:
                    doc = "".join(current_chunk).strip()
                    if doc:
                        final_chunks.append(doc)

                # 2. Smart Overlap Logic:
                # Remove items from the start of the list until the size is
                # small enough to allow "breathing room", but keeping the overlap.
                # We want (current_length) to drop to approx (chunk_overlap).

                # While current size is greater than allowed overlap,
                # remove the oldest sentence/word (FIFO).
                while current_length > self.chunk_overlap and current_chunk:
                    removed_part = current_chunk.pop(0)
                    current_length -= len(removed_part)

            # Add the new split to the current buffer
            current_chunk.append(split)
            current_length += split_len

        # Add the last remaining chunk
        if current_chunk:
            doc = "".join(current_chunk).strip()
            if doc:
                final_chunks.append(doc)

        return final_chunks
