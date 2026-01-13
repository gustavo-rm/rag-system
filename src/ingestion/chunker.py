import re
from typing import List


class Chunker:
    """
    Uma classe para dividir texto em chunks de forma inteligente e recursiva.
    Esta abordagem preserva a coesão semântica do texto ao tentar dividir
    por separadores lógicos (parágrafos, sentenças) antes de recorrer a
    separadores menos ideais. Também implementa sobreposição (overlap)
    entre os chunks para evitar a perda de contexto.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Inicializa o Chunker.

        Parâmetros:
        - chunk_size (int): O tamanho máximo de cada chunk em número de caracteres.
                            É crucial que este valor seja compatível com o limite
                            do seu modelo de embedding.
        - chunk_overlap (int): O número de caracteres de sobreposição entre chunks
                               consecutivos para garantir a continuidade do contexto.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("O chunk_overlap deve ser menor que o chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Lista de separadores, do mais ao menos semanticamente relevante.
        # Adicionados ? e ! para respeitar frases interrogativas/exclamativas
        # A ordem importa: primeiro parágrafos, depois frases, depois palavras.
        self.separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def _split_text_with_separators(self, text: str, separators: List[str]) -> List[str]:
        """
        Tenta dividir o texto usando a lista de separadores de forma recursiva.
        """
        final_chunks = []

        # Pega o primeiro separador da lista.
        separator = separators[0]
        # Pega os separadores restantes para a chamada recursiva.
        remaining_separators = separators[1:]

        # Se o separador for vazio, dividimos por caractere.
        if not separator:
            splits = list(text)
        else:
            # Usa uma expressão regular para manter o separador no final do split
            # re.escape garante que '.' ou '?' sejam lidos como texto, não comando regex
            splits = re.split(f"({re.escape(separator)})", text)
            splits = [s for s in splits if s]  # Remove strings vazias

            # Agrupa o texto e o separador
            merged_splits = []
            temp_split = ""
            for i in range(0, len(splits), 2):
                part = splits[i]
                sep = splits[i + 1] if i + 1 < len(splits) else ""
                merged_splits.append(part + sep)
            splits = [s for s in merged_splits if s]

        current_chunk = ""
        for s in splits:
            # Se um único split já é maior que o chunk_size,
            # chama a função recursivamente com os próximos separadores.
            if len(s) > self.chunk_size:
                if remaining_separators:
                    # A aplicação recursiva acontece aqui
                    final_chunks.extend(self._split_text_with_separators(s, remaining_separators))
                else:
                    # Se não há mais separadores, adicionamos o split "grande" mesmo assim.
                    final_chunks.append(s)

            # Se o split atual, somado ao chunk corrente, exceder o tamanho,
            # finalizamos o chunk corrente.
            elif len(current_chunk + s) > self.chunk_size:
                final_chunks.append(current_chunk)
                current_chunk = s

            # Senão, continuamos a construir o chunk corrente.
            else:
                current_chunk += s

        if current_chunk:
            final_chunks.append(current_chunk)

        return final_chunks

    def chunk_text(self, text: str) -> List[str]:
        """
        Método principal para dividir o texto em chunks com sobreposição.

        Parâmetros:
        - text (str): O texto de entrada a ser dividido.

        Retorna:
        - List[str]: Uma lista de chunks de texto.
        """
        # 1. Primeiro, fazemos uma divisão inicial recursiva para que nenhum
        #    elemento da lista seja maior que o chunk_size.
        initial_splits = self._split_text_with_separators(text, self.separators)

        # 2. Agora, agrupamos esses splits menores em chunks do tamanho desejado,
        #    respeitando a sobreposição.
        final_chunks = []
        buffer = ""

        for chunk in initial_splits:
            # Se o buffer + o novo chunk for menor que o tamanho alvo, apenas adiciona
            if len(buffer) + len(chunk) <= self.chunk_size:
                buffer += chunk
            else:
                # Se exceder, finaliza o chunk atual
                # .strip() evita salvar chunks cheios de espaços vazios nas pontas
                if buffer.strip():
                    final_chunks.append(buffer.strip())

                # O novo buffer começa com a sobreposição do chunk anterior
                # e o chunk atual.
                overlap_start = max(0, len(buffer) - self.chunk_overlap)
                buffer = buffer[overlap_start:] + chunk

        if buffer and buffer.strip():
            final_chunks.append(buffer.strip())

        return final_chunks
