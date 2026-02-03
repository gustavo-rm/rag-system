from src.config import Config

class _PromptsMeta(type):
    """Metaclass to allow dynamic property access on the Prompts class itself."""

    @property
    def CHATBOT_SYSTEM_PROMPT(cls): return cls.get("CHATBOT_SYSTEM_PROMPT")
    @property
    def CHATBOT_CONTEXT_TEMPLATE(cls): return cls.get("CHATBOT_CONTEXT_TEMPLATE")
    @property
    def HYDE_SYSTEM_PROMPT(cls): return cls.get("HYDE_SYSTEM_PROMPT")
    @property
    def HYDE_PROMPT_TEMPLATE(cls): return cls.get("HYDE_PROMPT_TEMPLATE")
    @property
    def MULTI_QUERY_SYSTEM_PROMPT(cls): return cls.get("MULTI_QUERY_SYSTEM_PROMPT")
    @property
    def MULTI_QUERY_PROMPT_TEMPLATE(cls): return cls.get("MULTI_QUERY_PROMPT_TEMPLATE")
    @property
    def ROUTER_SYSTEM_PROMPT(cls): return cls.get("ROUTER_SYSTEM_PROMPT")
    @property
    def ROUTER_CRITERIA_NOOP(cls): return cls.get("ROUTER_CRITERIA_NOOP")
    @property
    def ROUTER_CRITERIA_HYDE(cls): return cls.get("ROUTER_CRITERIA_HYDE")
    @property
    def ROUTER_CRITERIA_MULTI_QUERY(cls): return cls.get("ROUTER_CRITERIA_MULTI_QUERY")
    @property
    def ROUTER_PROMPT_TEMPLATE(cls): return cls.get("ROUTER_PROMPT_TEMPLATE")
    @property
    def RAG_SYSTEM_PROMPT(cls): return cls.get("RAG_SYSTEM_PROMPT")
    @property
    def RAG_USER_PROMPT_TEMPLATE(cls): return cls.get("RAG_USER_PROMPT_TEMPLATE")
    @property
    def SYNTHETIC_GEN_SYSTEM_PROMPT(cls): return cls.get("SYNTHETIC_GEN_SYSTEM_PROMPT")
    @property
    def SYNTHETIC_GEN_PROMPT_TEMPLATE(cls): return cls.get("SYNTHETIC_GEN_PROMPT_TEMPLATE")

class Prompts(metaclass=_PromptsMeta):
    """
    Centralized repository for all LLM prompts used in the RAG system.
    Prompts are organized by module/functional area and language.
    Accessing attributes (e.g. Prompts.CHATBOT_SYSTEM_PROMPT) will dynamically
    return the string in the configured language.
    """

    _PROMPTS = {
        "en-US": {
            "CHATBOT_SYSTEM_PROMPT": "You are an expert assistant in rewriting questions for search systems.",
            "CHATBOT_CONTEXT_TEMPLATE": """
        Conversation History:
        {chat_history}

        Last User Question: "{question}"

        You are a QUESTION REWRITING module.
        You do NOT answer questions.
        You ONLY rewrite the 'Last Question', strictly following the rules below.

        ==============================
        PRIORITY ORDER (FOLLOW STRICTLY)
        ==============================
        1. NEVER answer the question.
        2. If the question is clear and independent of the history, REPEAT IT EXACTLY as is.
        3. Use the history ONLY to replace ambiguous pronouns.
        4. If there is a topic change, IGNORE the history completely.

        ==============================
        NEW TOPIC DEFINITION
        ==============================
        Consider a NEW TOPIC when the main noun of the question changes
        relative to the previous history.

        Example:
        History: "What is the capital of Brazil?"
        Last Question: "And the relief?"
        → New topic → ignore history.

        ==============================
        MANDATORY RULES
        ==============================
        - Do NOT add information.
        - Do NOT rephrase style, tone, or vocabulary.
        - Do NOT make the question more specific.
        - Do NOT infer hidden intentions.
        - Do NOT connect ideas that are not explicitly in the question.

        ==============================
        REWRITING RULES
        ==============================
        - If the question uses pronouns ("he", "she", "it", "that") referring to the history,
        replace ONLY with the correct term already present in the history.
        - If there are NO ambiguous pronouns, REPEAT the question exactly,
        keeping all words, punctuation, and order.

        ==============================
        EXAMPLES
        ==============================

        Example 1 — Clear question (independent of history)
        Last Question:
        "What is the highest peak in Brazil?"

        Output:
        "What is the highest peak in Brazil?"

        ------------------------------

        Example 2 — Use of history-dependent pronoun
        History:
        "What is the capital of Brazil?"

        Last Question:
        "And what is its population?"

        Output:
        "What is the population of the capital of Brazil?"

        ------------------------------

        Example 3 — New topic (history ignored)
        History:
        "What is the capital of Brazil?"

        Last Question:
        "How is the relief?"

        Output:
        "How is the relief?"

        ==============================
        OUTPUT FORMAT
        ==============================
        - Return ONLY the final question.
        - Do NOT include explanations, comments, or any other text.

        Reformulated Question (text only):
        """,
            "HYDE_SYSTEM_PROMPT": "You are a synthetic data generator for RAG.",
            "HYDE_PROMPT_TEMPLATE": """
        Write a brief technical excerpt that answers the question below.
        Do not answer the question directly, but simulate what the text in a technical manual containing the answer would look like.
        Question: {question}
        Manual passage:
        """,
            "MULTI_QUERY_SYSTEM_PROMPT": "Generator of search variations in English.",
            "MULTI_QUERY_PROMPT_TEMPLATE": """
            You are an AI assistant expert in geographical and factual searches in ENGLISH.
            Your task is to generate {num} variations of the user's question to find the answer in technical documents.

            MANDATORY Rules:
            1. Answer ONLY in ENGLISH.
            2. Use technical synonyms. (Ex: "biggest mountain" -> "highest point", "highest peak", "maximum altitude").
            3. Do NOT answer the question. Just rewrite the variations.
            4. Do NOT write introductions like "Here are the variations". Return ONLY the questions, one per line.

            Original Question: "{question}"
            """,
            "ROUTER_SYSTEM_PROMPT": "You are a search intent classifier (RAG Router).",
            "ROUTER_CRITERIA_NOOP": "Use ONLY for extremely specific questions containing exact identifiers (IDs, Codes, Logs) that do not need expansion.",
            "ROUTER_CRITERIA_HYDE": "Use for complex, abstract questions, 'How does it work', 'Why', or theoretical definitions requiring reasoning.",
            "ROUTER_CRITERIA_MULTI_QUERY": "THE BEST OPTION for short, factual questions ('What is...', 'Who was...'), geographical, or vague queries. Use whenever synonyms are possible.",
            "ROUTER_PROMPT_TEMPLATE": """
        Analyze the USER QUESTION and classify it into one of the following search strategies:

        1. 'noop': {criteria_noop}
        2. 'hyde': {criteria_hyde}
        3. 'multi_query': {criteria_multi_query}

        EXAMPLES TO GUIDE YOUR DECISION:
        - "Error 500 on endpoint /login" -> noop
        - "What is the ID of client 9988?" -> noop
        - "Explain the impact of inflation on interest rates" -> hyde
        - "How does photosynthesis work?" -> hyde
        - "Capital of Brazil" -> multi_query
        - "What is the highest Brazilian peak?" -> multi_query (Geographical fact/Synonyms)
        - "Best beaches in the northeast" -> multi_query

        USER QUESTION: "{question}"

        Return ONLY the strategy name (noop, hyde or multi_query). Nothing else.
        Response:
        """,
            "RAG_SYSTEM_PROMPT": """
        You are a precise geographical and technical assistant.
        Your only source of truth are the [CONTEXTS] provided below.

        Guidelines:
        1. Answer the user's question using ONLY the information from the context.
        2. Answer in English in a fluid and direct manner.
        3. If the context contains the answer, explain it in detail.
        4. If the context mentions the subject but doesn't have the exact answer, say what you found about the topic.
        5. ONLY if the context is totally irrelevant, say: "The information was not found in the provided documents."
        """,
            "RAG_USER_PROMPT_TEMPLATE": """
        [RETRIEVED CONTEXTS]
        {context_block}

        [USER QUESTION]
        {question}

        Based strictly on the contexts above, what is the answer?
        """,
            "SYNTHETIC_GEN_SYSTEM_PROMPT": "You are an expert in creating AI training datasets.",
            "SYNTHETIC_GEN_PROMPT_TEMPLATE": """
        Below is an excerpt from a technical document.
        Your task: Write a SHORT and OBJECTIVE question that can be answered EXCLUSIVELY with the information in this excerpt.

        [EXCERPT]
        {chunk}
        [/EXCERPT]

        Answer ONLY the question. Do not add "Here is the question" or quotes.
        Question:
        """
        },
        "pt-BR": {
            "CHATBOT_SYSTEM_PROMPT": "Você é um assistente especialista em reescrever perguntas para sistemas de busca.",
            "CHATBOT_CONTEXT_TEMPLATE": """
        Histórico da Conversa:
        {chat_history}

        Última Pergunta do Usuário: "{question}"

        Você é um módulo de REESCRITA DE PERGUNTAS.
        Você NÃO responde perguntas.
        Você APENAS reescreve a 'Última Pergunta', seguindo rigorosamente as regras abaixo.

        ==============================
        ORDEM DE PRIORIDADE (SIGA ESTRITAMENTE)
        ==============================
        1. NUNCA responda à pergunta.
        2. Se a pergunta for clara e independente do histórico, REPITA-A EXATAMENTE como está.
        3. Use o histórico SOMENTE para substituir pronomes ambíguos.
        4. Se houver mudança de tópico, IGNORE completamente o histórico.

        ==============================
        DEFINIÇÃO DE NOVO TÓPICO
        ==============================
        Considere que há NOVO TÓPICO quando o substantivo principal da pergunta muda
        em relação ao histórico anterior.

        Exemplo:
        Histórico: "Qual é a capital do Brasil?"
        Última Pergunta: "E o relevo?"
        → Novo tópico → ignore o histórico.

        ==============================
        REGRAS OBRIGATÓRIAS
        ==============================
        - NÃO adicione informações.
        - NÃO reformule estilo, tom ou vocabulário.
        - NÃO torne a pergunta mais específica.
        - NÃO infira intenções ocultas.
        - NÃO conecte ideias que não estejam explicitamente na pergunta.

        ==============================
        REGRAS DE REESCRITA
        ==============================
        - Se a pergunta usar pronomes ("ele", "ela", "isso", "aquilo") referindo-se ao histórico,
        substitua APENAS pelo termo correto já presente no histórico.
        - Se NÃO houver pronomes ambíguos, REPITA a pergunta exatamente,
        mantendo todas as palavras, pontuação e ordem.

        ==============================
        EXEMPLOS
        ==============================

        Exemplo 1 — Pergunta clara (sem depender do histórico)
        Última Pergunta:
        "Qual o maior pico do Brasil?"

        Saída:
        "Qual o maior pico do Brasil?"

        ------------------------------

        Exemplo 2 — Uso de pronome dependente do histórico
        Histórico:
        "Qual é a capital do Brasil?"

        Última Pergunta:
        "E qual é a população dela?"

        Saída:
        "Qual é a população da capital do Brasil?"

        ------------------------------

        Exemplo 3 — Novo tópico (histórico ignorado)
        Histórico:
        "Qual é a capital do Brasil?"

        Última Pergunta:
        "Como é o relevo?"

        Saída:
        "Como é o relevo?"

        ==============================
        FORMATO DA SAÍDA
        ==============================
        - Retorne APENAS a pergunta final.
        - NÃO inclua explicações, comentários ou qualquer outro texto.

        Pergunta Reformulada (apenas o texto):
        """,
            "HYDE_SYSTEM_PROMPT": "Você é um gerador de dados sintéticos para RAG.",
            "HYDE_PROMPT_TEMPLATE": """
        Escreva um breve trecho técnico que responda à pergunta abaixo.
        Não responda a pergunta diretamente, mas simule como seria o texto em um manual técnico que contém a resposta.
        Pergunta: {question}
        Passagem do manual:
        """,
            "MULTI_QUERY_SYSTEM_PROMPT": "Gerador de variações de busca em Português.",
            "MULTI_QUERY_PROMPT_TEMPLATE": """
            Você é um assistente de IA especialista em buscas geográficas e factuais em PORTUGUÊS.
            Sua tarefa é gerar {num} variações da pergunta do usuário para encontrar a resposta em documentos técnicos.

            Regras OBRIGATÓRIAS:
            1. Responda APENAS em PORTUGUÊS DO BRASIL.
            2. Use sinônimos técnicos. (Ex: "maior montanha" -> "ponto culminante", "pico mais alto", "altitude máxima").
            3. NÃO responda à pergunta. Apenas reescreva as variações.
            4. NÃO escreva introduções como "Aqui estão as variações". Retorne APENAS as perguntas, uma por linha.

            Pergunta Original: "{question}"
            """,
            "ROUTER_SYSTEM_PROMPT": "Você é um classificador de intenção de busca (RAG Router).",
            "ROUTER_CRITERIA_NOOP": "Use APENAS para perguntas extremamente específicas que contenham identificadores exatos (CNPJ, IDs, Códigos, Logs) e que não precisem de expansão.",
            "ROUTER_CRITERIA_HYDE": "Use para perguntas complexas, abstratas, pedidos de 'Como funciona', 'Por que', ou definições teóricas que exigem raciocínio.",
            "ROUTER_CRITERIA_MULTI_QUERY": "A MELHOR OPÇÃO para perguntas curtas, factuais ('Qual é...', 'Quem foi...'), geográficas ou vagas. Use sempre que houver sinônimos possíveis.",
            "ROUTER_PROMPT_TEMPLATE": """
        Analise a PERGUNTA DO USUÁRIO e classifique-a em uma das seguintes estratégias de busca:

        1. 'noop': {criteria_noop}
        2. 'hyde': {criteria_hyde}
        3. 'multi_query': {criteria_multi_query}

        EXEMPLOS PARA GUIAR SUA DECISÃO:
        - "Erro 500 no endpoint /login" -> noop
        - "Qual o CPF do cliente 9988?" -> noop
        - "Explique o impacto da inflação nos juros" -> hyde
        - "Como funciona a fotossíntese?" -> hyde
        - "Capital do Brasil" -> multi_query
        - "Qual o maior pico brasileiro?" -> multi_query (Fato geográfico/Sinônimos)
        - "Melhores praias do nordeste" -> multi_query

        PERGUNTA DO USUÁRIO: "{question}"

        Retorne APENAS o nome da estratégia (noop, hyde ou multi_query). Nada mais.
        Resposta:
        """,
            "RAG_SYSTEM_PROMPT": """
        Você é um assistente geográfico e técnico preciso.
        Sua única fonte de verdade são os [CONTEXTOS] fornecidos abaixo.

        Diretrizes:
        1. Responda à pergunta do usuário usando APENAS as informações do contexto.
        2. Responda em Português do Brasil de forma fluida e direta.
        3. Se o contexto contiver a resposta, explique-a detalhadamente.
        4. Se o contexto mencionar o assunto mas não tiver a resposta exata, diga o que encontrou sobre o tema.
        5. SOMENTE se o contexto for totalmente irrelevante, diga: "A informação não foi encontrada nos documentos fornecidos."
        """,
            "RAG_USER_PROMPT_TEMPLATE": """
        [CONTEXTOS RECUPERADOS]
        {context_block}

        [PERGUNTA DO USUÁRIO]
        {question}

        Com base estritamente nos contextos acima, qual a resposta?
        """,
            "SYNTHETIC_GEN_SYSTEM_PROMPT": "Você é um especialista em criar datasets para treinamento de IA.",
            "SYNTHETIC_GEN_PROMPT_TEMPLATE": """
        Abaixo está um trecho de um documento técnico.
        Sua tarefa: Escreva uma pergunta CURTA e OBJETIVA que pode ser respondida EXCLUSIVAMENTE com as informações deste trecho.

        [TRECHO]
        {chunk}
        [/TRECHO]

        Responda APENAS a pergunta. Não adicione "Aqui está a pergunta" ou aspas.
        Pergunta:
        """
        }
    }

    @classmethod
    def get(cls, key: str) -> str:
        """Retrieves a prompt in the configured language, falling back to English."""
        lang = Config.LANGUAGE
        if lang not in cls._PROMPTS:
            lang = "en-US" # Default fallback

        return cls._PROMPTS.get(lang, {}).get(key, cls._PROMPTS["en-US"].get(key, ""))
