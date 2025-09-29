from typing import List, Dict


class ChatHistory:
    """Uma classe simples para armazenar e formatar o histórico da conversa."""

    def __init__(self):
        self.messages: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str):
        """Adiciona uma nova mensagem ao histórico.

        Args:
            role (str): O autor da mensagem ('user' ou 'assistant').
            content (str): O conteúdo da mensagem.
        """
        self.messages.append({"role": role, "content": content})

    def get_formatted_history(self) -> str:
        """Retorna o histórico formatado como uma única string."""
        if not self.messages:
            return "Esta é a primeira interação."

        formatted_string = "Histórico da Conversa Anterior:\n"
        for msg in self.messages:
            role = "Usuário" if msg["role"] == "user" else "Assistente"
            formatted_string += f"- {role}: {msg['content']}\n"
        return formatted_string.strip()