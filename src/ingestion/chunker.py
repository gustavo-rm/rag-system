import re
from typing import List


class Chunker:
    """
    Responsável por dividir textos longos em pedaços menores (chunks) preservando a coesão semântica.

    Utiliza uma abordagem recursiva baseada numa hierarquia de separadores
    (parágrafos -> sentenças -> palavras) para garantir que o texto não seja cortado
    no meio de uma ideia importante.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Inicializa o Chunker com configurações de tamanho e sobreposição.

        Args:
            chunk_size (int): O tamanho máximo de caracteres permitido por chunk.
                              Deve ser ajustado conforme o limite de tokens do modelo de Embedding.
            chunk_overlap (int): Quantidade de caracteres que se repetem entre o final de um chunk
                                 e o início do próximo. Garante continuidade de contexto.

        Raises:
            ValueError: Se o chunk_overlap for maior ou igual ao chunk_size.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("O chunk_overlap deve ser menor que o chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Hierarquia de separadores: tenta quebrar por parágrafo, depois por frase, depois por palavra.
        self.separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def _split_text_with_separators(self, text: str, separators: List[str]) -> List[str]:
        """
        (Método Interno) Tenta dividir o texto recursivamente usando a lista de separadores.

        Args:
            text (str): O texto a ser dividido.
            separators (List[str]): Lista de separadores restantes a serem tentados.

        Returns:
            List[str]: Lista de segmentos de texto que respeitam o tamanho máximo.
        """
        final_chunks = []
        separator = separators[0]
        remaining_separators = separators[1:]

        if not separator:
            splits = list(text)
        else:
            # Uso de re.escape para evitar que caracteres como '.' ou '?' quebrem o Regex.
            splits = re.split(f"({re.escape(separator)})", text)
            splits = [s for s in splits if s]

            # Reagrupa o separador com o texto anterior (ex: "Olá" + "." -> "Olá.")
            merged_splits = []
            for i in range(0, len(splits), 2):
                part = splits[i]
                sep = splits[i + 1] if i + 1 < len(splits) else ""
                merged_splits.append(part + sep)
            splits = [s for s in merged_splits if s]

        current_chunk = ""
        for s in splits:
            if len(s) > self.chunk_size:
                if remaining_separators:
                    final_chunks.extend(self._split_text_with_separators(s, remaining_separators))
                else:
                    final_chunks.append(s)

            elif len(current_chunk + s) > self.chunk_size:
                final_chunks.append(current_chunk)
                current_chunk = s
            else:
                current_chunk += s

        if current_chunk:
            final_chunks.append(current_chunk)

        return final_chunks

    def chunk_text(self, text: str) -> List[str]:
        """
        Processa o texto completo e retorna a lista final de chunks com sobreposição (overlap).

        Args:
            text (str): O texto bruto extraído do documento.

        Returns:
            List[str]: Lista de strings limpas e dimensionadas, prontas para embedding.
        """
        initial_splits = self._split_text_with_separators(text, self.separators)

        final_chunks = []
        buffer = ""

        for chunk in initial_splits:
            if len(buffer) + len(chunk) <= self.chunk_size:
                buffer += chunk
            else:
                # Aplica .strip() antes de salvar para evitar chunks que começam/terminam com espaços inúteis
                if buffer.strip():
                    final_chunks.append(buffer.strip())

                overlap_start = max(0, len(buffer) - self.chunk_overlap)
                buffer = buffer[overlap_start:] + chunk

        if buffer and buffer.strip():
            final_chunks.append(buffer.strip())

        return final_chunks