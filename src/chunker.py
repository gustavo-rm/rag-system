import nltk
import spacy


class Chunker:
    def __init__(self, method='sentences', chunk_size=100):
        """
        Inicializa a classe Chunker com o método de chunking e o tamanho do chunk.

        Parâmetros:
        - method: Método de chunking. Pode ser 'sentences' (dividir por sentenças),
                  'paragraphs' (dividir por parágrafos) ou 'tokens' (dividir por tokens).
        - chunk_size: Tamanho máximo do chunk (em número de caracteres ou tokens, dependendo do método).

        Se o método for 'tokens', o modelo SpaCy será carregado para processamento de tokens.
        Se o método for 'sentences', o NLTK será utilizado para tokenização de sentenças.
        """
        self.method = method
        self.chunk_size = chunk_size
        if method == 'tokens':
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("Baixando modelo do spaCy...")
                from spacy.cli import download
                download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
        elif method == 'sentences':
            nltk.download('punkt')

    def chunk_text(self, text):
        """
        Divide o texto de entrada em chunks com base no método especificado.

        Parâmetros:
        - text: Texto de entrada a ser dividido.

        Retorna:
        - Uma lista de chunks do texto.

        O método de chunking pode ser baseado em sentenças, parágrafos ou tokens.
        """
        if self.method == 'sentences':
            return self._chunk_by_sentences(text)
        elif self.method == 'paragraphs':
            return self._chunk_by_paragraphs(text)
        elif self.method == 'tokens':
            return self._chunk_by_tokens(text)
        else:
            raise ValueError("Método de chunking inválido.")

    def _chunk_by_sentences(self, text):
        """
        Divide o texto de entrada em chunks com base em sentenças.

        Parâmetros:
        - text: Texto de entrada a ser dividido.

        Retorna:
        - Uma lista de chunks, onde cada chunk contém uma ou mais sentenças, respeitando o tamanho máximo do chunk.
        """
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence + " "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def _chunk_by_paragraphs(self, text):
        """
        Divide o texto de entrada em chunks com base em parágrafos.

        Parâmetros:
        - text: Texto de entrada a ser dividido.

        Retorna:
        - Uma lista de chunks, onde cada chunk contém um ou mais parágrafos, respeitando o tamanho máximo do chunk.
        """
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) <= self.chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def _chunk_by_tokens(self, text):
        """
        Divide o texto de entrada em chunks com base em tokens, utilizando o SpaCy para tokenização.

        Parâmetros:
        - text: Texto de entrada a ser dividido.

        Retorna:
        - Uma lista de chunks, onde cada chunk contém um conjunto de tokens, respeitando o tamanho máximo do chunk.
        """
        doc = self.nlp(text)
        chunks = []
        current_chunk = []
        current_token_count = 0
        for sent in doc.sents:
            sentence_length = len(sent)
            if current_token_count + sentence_length <= self.chunk_size:
                current_chunk.append(sent.text)
                current_token_count += sentence_length
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sent.text]
                current_token_count = sentence_length
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks
