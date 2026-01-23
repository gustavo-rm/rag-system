import json
import os
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class ChatHistory:
    """
    Gerencia o histórico de conversação com estratégias de Janela Deslizante (Sliding Window)
    e Persistência.

    Evita o estouro de contexto do LLM mantendo apenas as mensagens mais recentes
    e permite salvar/carregar o estado da conversa.
    """

    def __init__(self, max_history_len: int = 10, persist_directory: str = "data/chat_logs"):
        """
        Inicializa o gerenciador de histórico.

        Args:
            max_history_len (int): Número máximo de TROCAS de mensagens (pares User/AI) a manter.
                                   Ex: 10 significa manter as últimas 20 mensagens (10 do user, 10 da AI).
            persist_directory (str): Pasta onde os históricos serão salvos.
        """
        self.max_history_len = max_history_len
        self.persist_directory = persist_directory
        self.messages: List[Dict[str, str]] = []

        # Garante que o diretório de logs existe
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)

    def is_empty(self) -> bool:
        """Retorna True se o histórico não tiver nenhuma mensagem."""
        return len(self.messages) == 0

    def add_message(self, role: str, content: str):
        """
        Adiciona uma nova mensagem e aplica a poda automática (trimming).

        Args:
            role (str): 'user', 'assistant' ou 'system'.
            content (str): O texto da mensagem.
        """
        self.messages.append({"role": role, "content": content})

        # Verifica se precisa podar o histórico antigo
        self._enforce_window_limit()

    def _enforce_window_limit(self):
        """
        (Interno) Mantém o histórico dentro do tamanho limite.

        Estratégia: Remove as mensagens mais antigas, MAS preserva a primeira
        se for uma mensagem de 'system' (instrução inicial), pois ela define a persona.
        """
        # Se temos uma mensagem de sistema no índice 0, não queremos removê-la
        has_system_prompt = len(self.messages) > 0 and self.messages[0]['role'] == 'system'

        # O limite efetivo é (2 * max_len) porque contamos pares
        effective_limit = self.max_history_len * 2

        if has_system_prompt:
            # Se exceder, mantemos a [0] e pegamos as últimas (limit - 1)
            if len(self.messages) > effective_limit + 1:
                removed_count = len(self.messages) - (effective_limit + 1)
                # Mantém system + últimas N
                self.messages = [self.messages[0]] + self.messages[-(effective_limit):]
                logger.debug(f"Histórico podado. {removed_count} mensagens antigas removidas (System prompt mantido).")
        else:
            # Poda simples
            if len(self.messages) > effective_limit:
                self.messages = self.messages[-effective_limit:]
                logger.debug("Histórico podado (janela deslizante aplicada).")

    def get_formatted_history(self, format_type: str = "text") -> str:
        """
        Retorna o histórico formatado para injetar no Prompt do LLM.

        Args:
            format_type (str): 'text' para string simples ou 'chatML' (futuro).

        Returns:
            str: Histórico formatado.
        """
        if not self.messages:
            return ""

        formatted_string = ""
        for msg in self.messages:
            # Pula mensagens de sistema na formatação visual, se desejar
            if msg["role"] == "system":
                continue

            role_name = "Usuário" if msg["role"] == "user" else "Assistente"
            formatted_string += f"{role_name}: {msg['content']}\n"

        return formatted_string.strip()

    def get_messages_for_api(self) -> List[Dict[str, str]]:
        """
        Retorna a lista crua de dicionários, ideal para APIs que aceitam
        o formato messages=[...] (como OpenAI e HuggingFace Chat Templates).
        """
        return self.messages

    def clear(self):
        """Limpa o histórico atual."""
        self.messages = []
        logger.info("Histórico de conversa limpo.")

    # --- Persistência (Salvar/Carregar) ---

    def save_session(self, session_id: str):
        """Salva a conversa atual em um arquivo JSON."""
        filepath = os.path.join(self.persist_directory, f"{session_id}.json")
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.messages, f, ensure_ascii=False, indent=4)
            logger.info(f"Sessão salva em: {filepath}")
        except Exception as e:
            logger.error(f"Erro ao salvar sessão: {e}")

    def load_session(self, session_id: str):
        """Carrega uma conversa anterior."""
        filepath = os.path.join(self.persist_directory, f"{session_id}.json")
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    self.messages = json.load(f)
                logger.info(f"Sessão carregada: {session_id} ({len(self.messages)} mensagens)")
            except Exception as e:
                logger.error(f"Erro ao carregar sessão: {e}")
        else:
            logger.warning(f"Sessão não encontrada: {session_id}")
