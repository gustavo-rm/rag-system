import language_tool_python

class QueryCorrector:
    """
    Uma classe para pré-processar e corrigir a entrada do usuário usando
    a poderosa ferramenta LanguageTool, com suporte completo ao português.
    """
    def __init__(self, language: str = 'pt-BR'):
        """
        Inicializa o corretor ortográfico e gramatical.

        Args:
            language (str): O código do idioma (ex: 'pt-BR').
        """
        try:
            # A biblioteca irá baixar o servidor do LanguageTool na primeira execução.
            # Isso pode levar um momento.
            self.tool = language_tool_python.LanguageTool(language)
            print(f"Corretor Ortográfico e Gramatical (LanguageTool) inicializado para o idioma: '{language}'.")
        except Exception as e:
            print(f"ERRO: Não foi possível inicializar o LanguageTool. Verifique se o Java está instalado. Erro: {e}")
            self.tool = None

    def correct(self, text: str) -> str:
        """
        Corrige erros de digitação e gramática em uma string de texto.

        Args:
            text (str): A pergunta original do usuário.

        Returns:
            A pergunta corrigida.
        """
        if not self.tool:
            print("AVISO: LanguageTool não está disponível. Retornando o texto original.")
            return text

        # O método tool.correct() aplica todas as regras de correção
        corrected_text = self.tool.correct(text)
        return corrected_text