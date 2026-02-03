class Prompts:
    """
    Centralized repository for all LLM prompts used in the RAG system.
    Prompts are organized by module/functional area.
    """

    # --- Chatbot Prompts ---
    CHATBOT_SYSTEM_PROMPT = "You are an expert assistant in rewriting questions for search systems."

    CHATBOT_CONTEXT_TEMPLATE = """
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
        """

    # --- Query Transformer Prompts ---

    # HyDE (Hypothetical Document Embeddings)
    HYDE_SYSTEM_PROMPT = "You are a synthetic data generator for RAG."
    HYDE_PROMPT_TEMPLATE = """
        Write a brief technical excerpt that answers the question below.
        Do not answer the question directly, but simulate what the text in a technical manual containing the answer would look like.
        Question: {question}
        Manual passage:
        """

    # Multi-Query
    MULTI_QUERY_SYSTEM_PROMPT = "Generator of search variations in Portuguese."
    MULTI_QUERY_PROMPT_TEMPLATE = """
            You are an AI assistant expert in geographical and factual searches in PORTUGUESE.
            Your task is to generate {num} variations of the user's question to find the answer in technical documents.

            MANDATORY Rules:
            1. Answer ONLY in BRAZILIAN PORTUGUESE.
            2. Use technical synonyms. (Ex: "biggest mountain" -> "highest point", "highest peak", "maximum altitude").
            3. Do NOT answer the question. Just rewrite the variations.
            4. Do NOT write introductions like "Here are the variations". Return ONLY the questions, one per line.

            Original Question: "{question}"
            """

    # --- Router Prompts ---
    ROUTER_SYSTEM_PROMPT = "You are a search intent classifier (RAG Router)."
    ROUTER_CRITERIA_NOOP = "Use ONLY for extremely specific questions containing exact identifiers (IDs, Codes, Logs) that do not need expansion."
    ROUTER_CRITERIA_HYDE = "Use for complex, abstract questions, 'How does it work', 'Why', or theoretical definitions requiring reasoning."
    ROUTER_CRITERIA_MULTI_QUERY = "THE BEST OPTION for short, factual questions ('What is...', 'Who was...'), geographical, or vague queries. Use whenever synonyms are possible."

    ROUTER_PROMPT_TEMPLATE = """
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
        """

    # --- Pipeline (RAG Generation) Prompts ---
    RAG_SYSTEM_PROMPT = """
        You are a precise geographical and technical assistant.
        Your only source of truth are the [CONTEXTS] provided below.

        Guidelines:
        1. Answer the user's question using ONLY the information from the context.
        2. Answer in Brazilian Portuguese in a fluid and direct manner.
        3. If the context contains the answer, explain it in detail.
        4. If the context mentions the subject but doesn't have the exact answer, say what you found about the topic.
        5. ONLY if the context is totally irrelevant, say: "The information was not found in the provided documents."
        """

    RAG_USER_PROMPT_TEMPLATE = """
        [RETRIEVED CONTEXTS]
        {context_block}

        [USER QUESTION]
        {question}

        Based strictly on the contexts above, what is the answer?
        """

    # --- Synthetic Data Generation Prompts ---
    SYNTHETIC_GEN_SYSTEM_PROMPT = "You are an expert in creating AI training datasets."
    SYNTHETIC_GEN_PROMPT_TEMPLATE = """
        Below is an excerpt from a technical document.
        Your task: Write a SHORT and OBJECTIVE question that can be answered EXCLUSIVELY with the information in this excerpt.

        [EXCERPT]
        {chunk}
        [/EXCERPT]

        Answer ONLY the question. Do not add "Here is the question" or quotes.
        Question:
        """
