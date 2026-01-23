import re
from typing import List, Optional


class Chunker:
    """
    Responsável por dividir textos longos em pedaços menores (chunks) preservando a coesão semântica.

    Implementa uma estratégia de "Janela Deslizante" (Sliding Window) sobre unidades semânticas.
    O texto é primeiro quebrado recursivamente na maior unidade possível (Parágrafo -> Sentença -> Palavra)
    e depois reagrupado para preencher o tamanho do chunk, mantendo um overlap de contexto.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Inicializa o Chunker.

        Args:
            chunk_size (int): O tamanho máximo de caracteres desejado por chunk.
            chunk_overlap (int): Quantidade de caracteres que devem se repetir entre chunks adjacentes
                                 para manter o contexto.

        Raises:
            ValueError: Se o chunk_overlap for maior ou igual ao chunk_size.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("O chunk_overlap deve ser estritamente menor que o chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Hierarquia de separadores: Parágrafo duplo -> Parágrafo -> Sentença -> Pontuação -> Espaço -> Caractere
        self.separators = ["\n\n", "\n", ". ", "? ", "! ", ";", " ", ""]

    def chunk_text(self, text: str) -> List[str]:
        """
        Método principal para processar um texto bruto e retornar chunks otimizados.

        Realiza uma pré-limpeza no texto para normalizar espaços antes de segmentar.

        Args:
            text (str): O texto bruto a ser processado.

        Returns:
            List[str]: Lista de chunks de texto prontos para vetorização.
        """
        if not text:
            return []

        # 1. Sanitização básica: remove excesso de espaços em branco (tabulações, múltiplos espaços)
        # Isso evita criar "chunks fantasmas" ou desperdiçar tokens.
        text = re.sub(r'\s+', ' ', text).strip()

        # 2. Divisão Recursiva: Quebra o texto em unidades menores que o chunk_size
        splits = self._recursive_split(text, self.separators)

        # 3. Fusão com Overlap: Reagrupa as unidades em chunks finais
        return self._merge_splits(splits)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """
        Divide o texto recursivamente até que todos os pedaços sejam menores que o chunk_size.

        Args:
            text (str): O texto ou fragmento a ser analisado.
            separators (List[str]): Lista de separadores disponíveis para tentativa de quebra.

        Returns:
            List[str]: Uma lista plana de "unidades atômicas" (ex: sentenças ou palavras)
                       onde cada item é garantidamente menor ou igual ao chunk_size
                       (a menos que uma única palavra exceda o limite).
        """
        final_splits = []

        # Se o texto já cabe no chunk, não precisamos dividir mais.
        if len(text) <= self.chunk_size:
            return [text]

        # Se acabaram os separadores, somos forçados a retornar o texto como está
        # (caso raro de uma palavra gigante) ou dividir por caractere se o separador vazio estiver na lista.
        if not separators:
            return [text]

        separator = separators[0]
        next_separators = separators[1:]

        # Tenta dividir pelo separador atual
        if separator == "":
            # Caso base: dividir por caractere
            splits = list(text)
        else:
            # Mantém o separador no split.
            # O regex `({sep})` com parênteses captura o separador.
            _splits = re.split(f"({re.escape(separator)})", text)
            splits = []

            # Reagrupa o separador com o texto anterior para manter a pontuação correta.
            # Ex: ["Olá", ".", " Mundo"] -> ["Olá.", " Mundo"]
            for i in range(1, len(_splits), 2):
                splits.append(_splits[i - 1] + _splits[i])
            # Adiciona o último pedaço se sobrar (sem separador no final)
            if len(_splits) % 2 != 0:
                splits.append(_splits[-1])

        # Filtra strings vazias
        splits = [s for s in splits if s]

        # Avalia cada pedaço resultante
        for s in splits:
            if len(s) > self.chunk_size:
                # Se o pedaço ainda é grande, recurse com o PRÓXIMO separador
                final_splits.extend(self._recursive_split(s, next_separators))
            else:
                final_splits.append(s)

        return final_splits

    def _merge_splits(self, splits: List[str]) -> List[str]:
        """
        Combina pequenas unidades de texto em chunks maiores, gerenciando o overlap.

        Diferente da abordagem simples de string slicing, este método remove unidades inteiras
        do início do buffer para criar o overlap, garantindo que não cortamos palavras ao meio.

        Args:
            splits (List[str]): Lista de unidades de texto (ex: sentenças).

        Returns:
            List[str]: Lista final de chunks.
        """
        final_chunks = []
        current_chunk: List[str] = []
        current_length = 0

        for split in splits:
            split_len = len(split)

            # Verifica se adicionar o novo split excede o tamanho máximo
            if current_length + split_len > self.chunk_size:

                # 1. Salva o chunk atual se ele não estiver vazio
                if current_length > 0:
                    doc = "".join(current_chunk).strip()
                    if doc:
                        final_chunks.append(doc)

                # 2. Lógica de Overlap Inteligente:
                # Remove itens do início da lista até que o tamanho seja
                # pequeno o suficiente para permitir "respirar", mas mantendo o overlap.
                # Queremos que (current_length) caia para aprox (chunk_overlap).

                # Enquanto o tamanho atual for maior que o overlap permitido,
                # removemos a sentença/palavra mais antiga (FIFO).
                while current_length > self.chunk_overlap and current_chunk:
                    removed_part = current_chunk.pop(0)
                    current_length -= len(removed_part)

            # Adiciona o novo split ao buffer atual
            current_chunk.append(split)
            current_length += split_len

        # Adiciona o último chunk remanescente
        if current_chunk:
            doc = "".join(current_chunk).strip()
            if doc:
                final_chunks.append(doc)

        return final_chunks