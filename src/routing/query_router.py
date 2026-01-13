from ..components.llm import LLM

class QueryRouter:
    """
    Usa um LLM para analisar uma pergunta de usuário e decidir qual estratégia de
    transformação de consulta é a mais adequada.
    """

    def __init__(self, llm: LLM):
        self.llm = llm
        self.system_prompt = """Sua tarefa é agir como um roteador de ferramentas. Com base na pergunta do usuário, escolha a melhor palavra-chave da lista fornecida para otimizar a busca. Responda APENAS com a palavra-chave escolhida (ex: 'hyde'). Não adicione nenhuma outra palavra ou pontuação."""

        # As descrições e as palavras-chave de retorno
        self.tool_map = {
            "noop": {
                "class_name": "NoOpTransformer",
                "description": "Use para perguntas longas, detalhadas e muito específicas, onde adicionar mais informações poderia gerar ruído."
            },
            "hyde": {
                "class_name": "HyDETransformer",
                "description": "Use para perguntas que buscam definições, explicações factuais ou termos técnicos. Ela cria um documento ideal para a busca."
            },
            "multi_query": {
                "class_name": "MultiQueryTransformer",
                "description": "Use para perguntas curtas, ambíguas ou abertas. Ela cria múltiplas variações para cobrir diferentes ângulos da busca."
            }
        }

    def _format_prompt(self, question: str) -> str:
        """Formata o prompt para o LLM roteador."""
        prompt = f"""
        Pergunta do Usuário: "{question}"

        Palavras-chave das Ferramentas Disponíveis:
        """
        for keyword, tool_info in self.tool_map.items():
            prompt += f"- Palavra-chave: {keyword}\n  Descrição: {tool_info['description']}\n"

        prompt += "\nQual é a melhor palavra-chave para esta pergunta?"
        return prompt

    def select_transformer(self, question: str) -> str:
        """
        Seleciona a melhor classe de transformador para uma dada pergunta.
        Retorna o nome da classe do transformador.
        """
        prompt = self._format_prompt(question)
        print("INFO: Roteador decidindo a melhor estratégia de transformação...")

        llm_choice = self.llm.generate_response(
            prompt=prompt,
            system_prompt=self.system_prompt,
            max_new_tokens=10,  # Apenas uma palavra-chave
            temperature=0.0
        ).strip().lower()  # Converte para minúsculas para facilitar a correspondência

        # Análise flexível: verifica qual palavra-chave está contida na resposta do LLM
        for keyword, tool_info in self.tool_map.items():
            if keyword in llm_choice:
                print(f"INFO: Roteador escolheu a estratégia: {tool_info['class_name']} (via keyword '{keyword}')")
                return tool_info['class_name']

        print(
            f"AVISO: Roteador retornou uma escolha inválida ou não reconhecida ('{llm_choice}'). Usando 'NoOpTransformer' como padrão.")
        return self.tool_map['noop']['class_name']