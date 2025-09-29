from ..components.llm import LLM


class QueryRouter:
    """
    Usa um LLM para analisar uma pergunta de usuário e decidir qual estratégia de
    transformação de consulta é a mais adequada.
    """

    def __init__(self, llm: LLM):
        self.llm = llm
        self.system_prompt = "Sua tarefa é agir como um roteador. Com base na pergunta do usuário, escolha a melhor ferramenta da lista fornecida para otimizar a busca. Responda APENAS com o nome da ferramenta escolhida (ex: 'HyDETransformer')."

        # As descrições das ferramentas são cruciais para a decisão do LLM
        self.tool_descriptions = {
            "NoOpTransformer": "Use esta ferramenta para perguntas longas, detalhadas e muito específicas, onde adicionar mais informações poderia gerar ruído.",
            "HyDETransformer": "Use esta ferramenta para perguntas que buscam definições, explicações factuais ou termos técnicos. Ela cria um documento ideal para a busca.",
            "MultiQueryTransformer": "Use esta ferramenta para perguntas curtas, ambíguas ou abertas. Ela cria múltiplas variações para cobrir diferentes ângulos da busca."
        }

    def _format_prompt(self, question: str) -> str:
        """Formata o prompt para o LLM roteador."""
        prompt = f"""
        Pergunta do Usuário: "{question}"

        Ferramentas Disponíveis:
        """
        for name, desc in self.tool_descriptions.items():
            prompt += f"- Nome: {name}\n  Descrição: {desc}\n"

        prompt += "\nQual é a melhor ferramenta para esta pergunta?"
        return prompt

    def select_transformer(self, question: str) -> str:
        """
        Seleciona a melhor classe de transformador para uma dada pergunta.

        Retorna:
            O nome da classe do transformador como uma string.
        """
        prompt = self._format_prompt(question)

        print("INFO: Roteador decidindo a melhor estratégia de transformação...")

        llm_choice = self.llm.generate_response(
            prompt=prompt,
            system_prompt=self.system_prompt,
            max_new_tokens=20,  # O nome da classe é curto
            temperature=0.0  # Queremos uma decisão determinística
        ).strip()

        # Validação simples para garantir que o LLM retornou uma ferramenta válida
        if llm_choice in self.tool_descriptions:
            print(f"INFO: Roteador escolheu a ferramenta: {llm_choice}")
            return llm_choice
        else:
            print(
                f"AVISO: Roteador retornou uma escolha inválida ('{llm_choice}'). Usando 'NoOpTransformer' como padrão.")
            return "NoOpTransformer"