import fitz  # PyMuPDF
import os
import re
import logging
# Configuração de Logger
logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    Processa arquivos PDF para extrair e limpar texto, além de extrair imagens.
    Utiliza a biblioteca PyMuPDF (fitz) para performance e precisão superiores.
    """

    def __init__(self, pdf_path: str):
        """
        Inicializa o processador com o caminho do arquivo PDF.

        Parâmetros:
        - pdf_path (str): Caminho do arquivo PDF a ser processado.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"O arquivo PDF não foi encontrado em: {pdf_path}")
        self.pdf_path = pdf_path

    def _clean_text(self, text: str) -> str:
        """
        Realiza uma limpeza básica no texto extraído para melhorar a qualidade.
        - Junta palavras que foram quebradas por hífens.
        - Remove quebras de linha excessivas.
        - Normaliza espaços em branco.
        """
        # 1. Junta palavras que foram separadas por hífen no final da linha
        # Ex: "inteli- gência" -> "inteligência"
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)

        # 2. Substitui múltiplos espaços ou quebras de linha por um único espaço
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def extract_text(self, clean: bool = True) -> str:
        """
        Extrai todo o texto de um arquivo PDF usando PyMuPDF para alta fidelidade.

        Parâmetros:
        - clean (bool): Se True, aplica a função de limpeza ao texto extraído.

        Retorna:
        - Uma string contendo todo o texto extraído do PDF.
        """
        full_text = ""
        try:
            with fitz.open(self.pdf_path) as doc:
                for page in doc:
                    full_text += page.get_text() + " "
        except Exception as e:
            logger.error(f"Erro ao processar o PDF {self.pdf_path}: {e}")
            return ""

        if clean:
            return self._clean_text(full_text)

        return full_text.strip()

    def extract_images(self, output_folder: str = 'imagens_extraidas'):
        """
        Extrai todas as imagens de um arquivo PDF.
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        image_count = 0
        with fitz.open(self.pdf_path) as doc:
            for page_num in range(len(doc)):
                for img_index, img in enumerate(doc.get_page_images(page_num)):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    image_filename = os.path.join(output_folder, f"image_{page_num + 1}_{img_index + 1}.{image_ext}")

                    with open(image_filename, "wb") as image_file:
                        image_file.write(image_bytes)
                    image_count += 1

        logger.info(f"Total de {image_count} imagens extraídas para a pasta '{output_folder}'.")
        return image_count


# --- Como Usar ---
if __name__ == '__main__':
    # Crie um arquivo PDF de exemplo chamado 'exemplo.pdf' para testar
    # ou aponte para um PDF existente.
    pdf_file_path = 'exemplo.pdf'  # Substitua pelo caminho do seu PDF
    if not os.path.exists(pdf_file_path):
        logger.warning(f"Arquivo de exemplo '{pdf_file_path}' não encontrado. Crie um para testar o código.")
    else:
        processor = PDFProcessor(pdf_path=pdf_file_path)

        # Extrai e limpa o texto
        logger.info("--- Extraindo Texto Limpo ---")
        cleaned_text = processor.extract_text()
        logger.info(cleaned_text[:1000] + "...")  # Imprime os primeiros 1000 caracteres

        logger.info("\n" + "=" * 50 + "\n")

        # Extrai as imagens
        logger.info("--- Extraindo Imagens ---")
        processor.extract_images()