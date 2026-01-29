import logging
import numpy as np

# Component imports
from src.components.llm import LLM
from src.chat.history import ChatHistory
from src.caching.cache_manager import CacheManager
from src.caching.semantic_cache import SemanticCache
from src.pipeline import RAGSystem
from src.preprocessing.query_corrector import QueryCorrector

# Logger Configuration
logger = logging.getLogger(__name__)


class Chatbot:
    """
    Conversational controller that manages the User <-> RAG interaction flow.

    Responsibilities:
    1. Manage conversation history (Memory).
    2. Contextualize questions (Transform "And him?" into "Who is the author?").
    3. Check caches (Exact and Semantic) for fast responses.
    4. Trigger the RAGSystem to obtain document-based answers.
    """

    def __init__(self, llm: LLM, rag_system: RAGSystem,
                 cache_manager: CacheManager, semantic_cache: SemanticCache,
                 query_corrector: QueryCorrector):
        """
        Initializes the Chatbot.

        Args:
            llm (LLM): Used for the question contextualization (condensation) task.
            rag_system (RAGSystem): The search and response engine.
            cache_manager (CacheManager): Layer 1 Cache (Exact).
            semantic_cache (SemanticCache): Layer 2 Cache (Semantic).
            query_corrector (QueryCorrector): Tool for cleaning and spelling correction.
        """
        self.llm = llm
        self.rag_system = rag_system
        self.cache_manager = cache_manager
        self.semantic_cache = semantic_cache
        self.query_corrector = query_corrector

        # In-memory history
        self.history = ChatHistory(max_history_len=6)

        # --- CONTEXTUALIZATION PROMPT CONFIGURATION (SHIELDED) ---
        self.context_system_prompt = "You are an expert assistant in rewriting questions for search systems."
        self.context_prompt_template = self._get_context_prompt_template()

    @staticmethod
    def _get_context_prompt_template() -> str:
        """
        Returns the prompt template for question contextualization.
        This prompt contains "Guardrails" to avoid context hallucination.
        """
        return """
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

    def _contextualize_question(self, question: str) -> str:
        """
        Transforms a context-dependent question into a Standalone question,
        with protection against topic switching.

        Example 1 (Continuation):
        History: "Who discovered Brazil?" -> "Cabral".
        Question: "In what year?"
        Output: "In what year did Cabral discover Brazil?"

        Example 2 (Topic Change):
        History: "Who discovered Brazil?" -> "Cabral".
        Question: "What is the capital of France?"
        Output: "What is the capital of France?" (History ignored)

        Args:
            question (str): The user's raw question.

        Returns:
            str: The contextualized (standalone) question.
        """
        # 1. Optimization: If there is no history, the question is already standalone.
        if self.history.is_empty():
            return question

        # 2. Optimization: Long questions (>10 words) usually already contain the necessary context.
        if len(question.split()) > 10:
            logger.debug("Long question detected. Assuming it is standalone.")
            return question

        logger.info("Contextualizing question based on history...")

        chat_history_str = self.history.get_formatted_history()
        prompt = self.context_prompt_template.format(chat_history=chat_history_str, question=question)

        try:
            standalone_question = self.llm.generate_response(
                prompt=prompt,
                system_prompt=self.context_system_prompt,
                max_new_tokens=100,  # Short to avoid rambling
                temperature=0.2  # Low temperature to strictly follow rules
            )

            # Defensive cleanup of common prefixes
            clean_q = standalone_question.strip().replace('Reformulated Question:', '').replace('"', '')

            logger.info(f"🔄 Original Question: '{question}' | Standalone: '{clean_q}'")
            return clean_q

        except Exception as e:
            logger.error(f"Contextualization failed: {e}. Using original question.")
            return question

    def chat(self, user_input: str) -> str:
        """
        Processes the user message end-to-end.
        Flow: Correction -> Contextualization -> Cache -> RAG -> History.

        Args:
            user_input (str): The input message from the user.

        Returns:
            str: The assistant's response.
        """
        if not user_input.strip():
            return ""

        logger.info(f"💬 User: {user_input}")

        # --- 1. SPELLING CORRECTION ---
        corrected_input = self.query_corrector.correct_query(user_input)
        if corrected_input != user_input:
            logger.info(f"✏️ Correction applied: '{user_input}' -> '{corrected_input}'")

        # --- 2. CONTEXTUALIZATION ---
        standalone_question = self._contextualize_question(corrected_input)

        # --- 3. CACHE CHECK (L1 and L2) ---
        cached_response = self.cache_manager.get(standalone_question)
        if cached_response:
            logger.info("🚀 Cache L1 HIT (Exact)")
            self._update_history(user_input, cached_response)
            return cached_response

        # Prepare Embedding for Semantic Cache
        try:
            query_embedding_list = self.rag_system.embedder.generate_embeddings([standalone_question])
            query_embedding_np = np.array([query_embedding_list[0]], dtype='float32')

            cached_response = self.semantic_cache.check(query_embedding_np)
            if cached_response:
                logger.info("🚀 Cache L2 HIT (Semantic)")
                self.cache_manager.set(standalone_question, cached_response)
                self._update_history(user_input, cached_response)
                return cached_response

        except Exception as e:
            logger.warning(f"Error checking semantic cache (ignoring): {e}")
            query_embedding_np = None

        # --- 4. RAG EXECUTION ---
        logger.info("🔍 Cache MISS. Triggering RAG System...")

        try:
            rag_result = self.rag_system.ask(standalone_question, rerank_top_n=5)
            answer = rag_result['answer']
            strategy = rag_result.get('strategy_used', 'Unknown')
            logger.info(f"✅ Response generated via RAG (Strategy: {strategy})")

        except Exception as e:
            logger.error(f"❌ Critical error in RAGSystem: {e}")
            return "Sorry, I encountered an internal error processing your request."

        # --- 5. CACHE AND HISTORY UPDATE ---
        if answer and "not found" not in answer.lower():
            self.cache_manager.set(standalone_question, answer)
            if query_embedding_np is not None:
                self.semantic_cache.add(query_embedding_np, answer)

        self._update_history(user_input, answer)

        return answer

    def _update_history(self, user_msg: str, assistant_msg: str):
        """Helper to save to history while keeping the interface clean."""
        self.history.add_message(role="user", content=user_msg)
        self.history.add_message(role="assistant", content=assistant_msg)
