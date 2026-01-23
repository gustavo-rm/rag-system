import torch
import logging
from typing import Optional, Dict, Any
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedTokenizer
)

# Logger configuration
logger = logging.getLogger(__name__)


class LLMGenerationError(Exception):
    """Custom exception raised when any critical failure occurs in text generation."""
    pass


try:
    from openai import OpenAI, OpenAIError
except ImportError:
    OpenAI = None
    OpenAIError = Exception

# --- Defaults ---
DEFAULT_LOCAL_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_CONTEXT_WINDOW = 4000


class LLM:
    """
    Unified and optimized interface for text generation using Local LLMs (Hugging Face) or API (OpenAI).

    This class implements the Facade pattern to abstract the complexity of:
    1. Model loading (Local vs API).
    2. Memory management (Automatic 4-bit quantization).
    3. Compatibility (Tokenizer corrections and Remote Code).
    4. Resilience (Cache fallback in case of error).

    Attributes:
        method (str): The execution method ('local' or 'openai').
        model_name (str): The identifier of the model in use.
        context_window (int): The maximum token limit (input + output).
        device (str): Execution device ('cuda' or 'cpu').
    """

    def __init__(self, method: str = 'local', model_name: Optional[str] = None,
                 api_key: Optional[str] = None, context_window: int = DEFAULT_CONTEXT_WINDOW):
        """
        Initializes the LLM instance with automatic configuration based on the environment.

        Args:
            method (str): 'local' (GPU/CPU) or 'openai' (API).
            model_name (str, optional): Hugging Face or OpenAI ID. If None, uses DEFAULTs.
            api_key (str, optional): Required if method='openai'.
            context_window (int): Safety token limit.

        Raises:
            ImportError: If method='openai' and the lib is not installed.
            ValueError: If mandatory parameters (e.g., api_key) are missing.
            RuntimeError: If there is a critical failure loading the local model.
        """
        self.method = method
        self.model_name = model_name
        self.context_window = context_window
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialization routing
        if method == 'openai':
            self._setup_openai(api_key)
        elif method == 'local':
            self._setup_local_model()
        else:
            raise ValueError(f"Invalid method '{method}'. Choose 'local' or 'openai'.")

    def _setup_openai(self, api_key: str) -> None:
        """
        Configures the OpenAI client.

        Args:
            api_key (str): OpenAI authentication key.

        Raises:
            ImportError: If the `openai` library is missing.
            ValueError: If api_key is empty or null.
        """
        if not OpenAI:
            raise ImportError("Missing 'openai' library. Install with: pip install openai")
        if not api_key:
            raise ValueError("The 'api_key' parameter is required for the 'openai' method.")

        self.client = OpenAI(api_key=api_key)
        self.model_name = self.model_name or DEFAULT_OPENAI_MODEL
        logger.info(f"☁️ OpenAI LLM ready: {self.model_name}")

    def _setup_local_model(self) -> None:
        """
        Orchestrates local model loading, deciding quantization strategies
        and compatibility fixes dynamically.

        Raises:
            RuntimeError: If any error occurs during Tokenizer or Model loading (e.g., lack of VRAM, connection).
        """
        self.model_name = self.model_name or DEFAULT_LOCAL_MODEL
        logger.info(f"🖥️ Preparing Local LLM '{self.model_name}' on: {self.device.upper()}")

        if self.device == "cpu":
            logger.warning("⚠️ PERFORMANCE ALERT: Running LLM on CPU will be significantly slow.")

        try:
            # 1. Load Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._fix_tokenizer_padding()

            # 2. Build loading arguments (Factory Method)
            loader_kwargs = self._build_loader_kwargs()

            # 3. Load Model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **loader_kwargs  # Unpack dynamic arguments
            )

            logger.info("✅ Local model loaded successfully.")

        except Exception as e:
            logger.error(f"❌ Fatal error loading model: {e}")
            raise RuntimeError(f"Critical failure loading local model: {e}") from e

    # --- Configuration Helper Methods (Modularization) ---

    def _build_loader_kwargs(self) -> Dict[str, Any]:
        """
        Builds the configuration dictionary (kwargs) for `from_pretrained`.

        Isolates the decision logic:
        - If Unsloth/BNB: Do not pass quantization config (use native).
        - If Raw Model: Create BitsAndBytes config.
        - If Phi-3/Falcon: Activate trust_remote_code.

        Returns:
            Dict[str, Any]: A dictionary containing parameters like `device_map`, `quantization_config`, etc.
        """
        kwargs = {}

        # 1. Hardware Type Configuration
        if self.device == "cuda":
            # --- CRITICAL FIX FOR 8GB GPU ---
            # Forces the entire model onto GPU 0.
            # "auto" might try to offload pieces to CPU, which breaks BNB 4-bit models.
            kwargs["device_map"] = {"": 0}

            # Quantization Logic (GPU Exclusive)
            if self._is_pre_quantized():
                # Case 1: Unsloth/BNB (Ready to use)
                logger.info("⚡ Pre-quantized model detected. Using native config.")
            else:
                # Case 2: Raw Model (Need to compress now)
                logger.info("🔧 Applying 4-bit on-the-fly quantization...")
                kwargs["quantization_config"] = self._get_bnb_config()

        else:
            # Fallback for CPU (Slow, but functional for testing without BNB quantization)
            kwargs["device_map"] = "auto"

        # 2. Model Specific Configuration (Remote Code / Attention)
        if self._needs_remote_code():
            logger.warning(f"🛡️ Activating 'trust_remote_code' for {self.model_name}")
            kwargs["trust_remote_code"] = True

            # Specific fix for Phi-3 and new transformers
            if "Phi-3" in self.model_name:
                kwargs["attn_implementation"] = "eager"

        return kwargs

    def _needs_remote_code(self) -> bool:
        """
        Checks if the model requires remote code execution (e.g., new architectures).

        Returns:
            bool: True if the model is in the list of architectures requiring `trust_remote_code`.
        """
        keywords = ["Phi-3", "falcon", "mpt", "glm"]
        return any(k in self.model_name for k in keywords)

    def _is_pre_quantized(self) -> bool:
        """
        Detects if the model already has quantized weights.

        Returns:
            bool: True if the model name indicates quantization (bnb-4bit, awq, gptq).
        """
        # Unsloth uses 'bnb-4bit', others use 'awq', 'gptq'
        indicators = ["bnb-4bit", "awq", "gptq", "-quantized"]
        return any(i in self.model_name.lower() for i in indicators)

    def _get_bnb_config(self) -> BitsAndBytesConfig:
        """
        Generates default 4-bit (NF4) quantization configuration.

        Returns:
            BitsAndBytesConfig: Configuration object for injection into `from_pretrained`.
        """
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    def _fix_tokenizer_padding(self) -> None:
        """Ensures the tokenizer has a pad token (avoids generation errors)."""
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    # --- Text Generation ---

    def generate_response(self, prompt: str, system_prompt: str, max_new_tokens: int = 512,
                          temperature: float = 0.1, use_cache: bool = True) -> str:
        """
        Generates a textual response. Acts as a Dispatcher for Local or OpenAI.

        Args:
            prompt (str): User input.
            system_prompt (str): System instructions.
            max_new_tokens (int): Limit of tokens to generate.
            temperature (float): Creativity (0.0 to 1.0).
            use_cache (bool): Activates KV cache for speed (True) or deactivates for VRAM saving (False).

        Returns:
            str: The generated text, cleaned and without special tokens.

        Raises:
            LLMGenerationError: Wrapper for any exception that occurs during inference (network or local).
        """
        try:
            if self.method == 'openai':
                return self._generate_openai(prompt, system_prompt, max_new_tokens, temperature)
            elif self.method == 'local':
                return self._generate_local(prompt, system_prompt, max_new_tokens, temperature, use_cache)
        except Exception as e:
            logger.error(f"Generation error ({self.method}): {str(e)}")
            raise LLMGenerationError(f"Text generation failure: {str(e)}") from e

    def _generate_openai(self, prompt: str, system_prompt: str, max_tokens: int, temperature: float) -> str:
        """
        Executes generation via OpenAI API.

        Args:
            prompt (str): User prompt.
            system_prompt (str): System prompt.
            max_tokens (int): Max tokens.
            temperature (float): Temperature.

        Returns:
            str: API response content.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()

    def _generate_local(self, prompt: str, system_prompt: str, max_new_tokens: int, temperature: float,
                        use_cache: bool) -> str:
        """
        Executes local inference with cache compatibility error handling.

        Args:
            prompt (str): User prompt.
            system_prompt (str): System prompt.
            max_new_tokens (int): Max output tokens.
            temperature (float): Sampling temperature.
            use_cache (bool): If True, uses KV cache.

        Returns:
            str: Decoded text.

        Raises:
            ValueError: If total size (input + output) exceeds context window.
            AttributeError/RuntimeError: If transformers library failure occurs (rethrown after attempts).
        """
        # 1. Formats the prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)

        input_len = prompt_ids.shape[1]

        # 2. Window Verification
        if input_len + max_new_tokens > self.context_window:
            msg = f"Prompt too long ({input_len}). Limit: {self.context_window}"
            logger.error(msg)
            raise ValueError(msg)

        # 3. Generation with Retry (Fallback)
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    prompt_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True if temperature > 0 else False,
                    temperature=temperature if temperature > 0 else None,
                    top_p=0.9,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=use_cache
                )
        except AttributeError as e:
            # Captures 'seen_tokens' compatibility error (common in Phi-3 + new Transformers)
            if "seen_tokens" in str(e) and use_cache:
                logger.warning("⚠️ Cache Error detected. Attempting fallback with use_cache=False...")
                with torch.no_grad():
                    outputs = self.model.generate(
                        prompt_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=True if temperature > 0 else False,
                        temperature=temperature,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=False  # Deactivates cache on retry
                    )
            else:
                raise e  # Rethrow if it's another error

        # 4. Decoding
        generated_ids = outputs[0][input_len:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
