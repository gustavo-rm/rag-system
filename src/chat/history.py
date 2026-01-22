import json
import os
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class ChatHistory:
    """
    Manages conversation history with Sliding Window and Persistence strategies.

    Prevents LLM context overflow by keeping only the most recent messages
    and allows saving/loading the conversation state.
    """

    def __init__(self, max_history_len: int = 10, persist_directory: str = "data/chat_logs"):
        """
        Initializes the history manager.

        Args:
            max_history_len (int): Maximum number of message EXCHANGES (User/AI pairs) to keep.
                                   E.g., 10 means keeping the last 20 messages (10 user, 10 AI).
            persist_directory (str): Folder where histories will be saved.
        """
        self.max_history_len = max_history_len
        self.persist_directory = persist_directory
        self.messages: List[Dict[str, str]] = []

        # Ensures the logs directory exists
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)

    def is_empty(self) -> bool:
        """Returns True if the history has no messages."""
        return len(self.messages) == 0

    def add_message(self, role: str, content: str):
        """
        Adds a new message and applies automatic trimming.

        Args:
            role (str): 'user', 'assistant', or 'system'.
            content (str): The text of the message.
        """
        self.messages.append({"role": role, "content": content})

        # Checks if old history needs trimming
        self._enforce_window_limit()

    def _enforce_window_limit(self):
        """
        (Internal) Keeps the history within the size limit.

        Strategy: Removes the oldest messages, BUT preserves the first one
        if it is a 'system' message (initial instruction), as it defines the persona.
        """
        # If we have a system message at index 0, we don't want to remove it
        has_system_prompt = len(self.messages) > 0 and self.messages[0]['role'] == 'system'

        # The effective limit is (2 * max_len) because we count pairs
        effective_limit = self.max_history_len * 2

        if has_system_prompt:
            # If exceeded, keep [0] and take the last (limit - 1)
            if len(self.messages) > effective_limit + 1:
                removed_count = len(self.messages) - (effective_limit + 1)
                # Keep system + last N
                self.messages = [self.messages[0]] + self.messages[-(effective_limit):]
                logger.debug(f"History trimmed. {removed_count} old messages removed (System prompt kept).")
        else:
            # Simple trimming
            if len(self.messages) > effective_limit:
                self.messages = self.messages[-effective_limit:]
                logger.debug("History trimmed (sliding window applied).")

    def get_formatted_history(self, format_type: str = "text") -> str:
        """
        Returns the history formatted for injection into the LLM Prompt.

        Args:
            format_type (str): 'text' for simple string or 'chatML' (future).

        Returns:
            str: Formatted history.
        """
        if not self.messages:
            return ""

        formatted_string = ""
        for msg in self.messages:
            # Skip system messages in visual formatting, if desired
            if msg["role"] == "system":
                continue

            role_name = "User" if msg["role"] == "user" else "Assistant"
            formatted_string += f"{role_name}: {msg['content']}\n"

        return formatted_string.strip()

    def get_messages_for_api(self) -> List[Dict[str, str]]:
        """
        Returns the raw list of dictionaries, ideal for APIs that accept
        the messages=[...] format (like OpenAI and HuggingFace Chat Templates).

        Returns:
            List[Dict[str, str]]: The list of message dictionaries.
        """
        return self.messages

    def clear(self):
        """Clears the current history."""
        self.messages = []
        logger.info("Conversation history cleared.")

    # --- Persistence (Save/Load) ---

    def save_session(self, session_id: str):
        """
        Saves the current conversation to a JSON file.

        Args:
            session_id (str): The unique identifier for the session.
        """
        filepath = os.path.join(self.persist_directory, f"{session_id}.json")
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.messages, f, ensure_ascii=False, indent=4)
            logger.info(f"Session saved to: {filepath}")
        except Exception as e:
            logger.error(f"Error saving session: {e}")

    def load_session(self, session_id: str):
        """
        Loads a previous conversation.

        Args:
            session_id (str): The unique identifier for the session.
        """
        filepath = os.path.join(self.persist_directory, f"{session_id}.json")
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    self.messages = json.load(f)
                logger.info(f"Session loaded: {session_id} ({len(self.messages)} messages)")
            except Exception as e:
                logger.error(f"Error loading session: {e}")
        else:
            logger.warning(f"Session not found: {session_id}")
