import torch
import logging
from typing import Optional, Dict, Any
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedTokenizer
)

# Configuração de logging
logger = logging.getLogger(__name__)


class LLMGenerationError(Exception):
    """Exceção personalizada lançada quando ocorre qualquer falha crítica na geração de texto."""
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
    Interface unificada e otimizada para geração de texto usando LLMs Locais (Hugging Face) ou API (OpenAI).

    Esta classe implementa o padrão Facade para abstrair a complexidade de:
    1. Carregamento de modelos (Local vs API).
    2. Gerenciamento de memória (Quantização 4-bit automática).
    3. Compatibilidade (Correção de tokenizers e Remote Code).
    4. Resiliência (Fallback de cache em caso de erro).

    Attributes:
        method (str): O método de execução ('local' ou 'openai').
        model_name (str): O identificador do modelo em uso.
        context_window (int): O limite máximo de tokens (entrada + saída).
        device (str): Dispositivo de execução ('cuda' ou 'cpu').
    """

    def __init__(self, method: str = 'local', model_name: Optional[str] = None,
                 api_key: Optional[str] = None, context_window: int = DEFAULT_CONTEXT_WINDOW):
        """
        Inicializa a instância do LLM com configuração automática baseada no ambiente.

        Args:
            method (str): 'local' (GPU/CPU) ou 'openai' (API).
            model_name (str, optional): ID do Hugging Face ou OpenAI. Se None, usa DEFAULTs.
            api_key (str, optional): Obrigatório se method='openai'.
            context_window (int): Limite de tokens de segurança.

        Raises:
            ImportError: Se method='openai' e a lib não estiver instalada.
            ValueError: Se parâmetros obrigatórios (ex: api_key) estiverem faltando.
            RuntimeError: Se houver falha crítica ao carregar o modelo local.
        """
        self.method = method
        self.model_name = model_name
        self.context_window = context_window
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Roteamento de inicialização
        if method == 'openai':
            self._setup_openai(api_key)
        elif method == 'local':
            self._setup_local_model()
        else:
            raise ValueError(f"Método '{method}' inválido. Escolha 'local' ou 'openai'.")

    def _setup_openai(self, api_key: str) -> None:
        """
        Configura o cliente OpenAI.

        Args:
            api_key (str): Chave de autenticação da OpenAI.

        Raises:
            ImportError: Caso a biblioteca `openai` não esteja instalada.
            ValueError: Caso a api_key seja vazia ou nula.
        """
        if not OpenAI:
            raise ImportError("Biblioteca 'openai' ausente. Instale com: pip install openai")
        if not api_key:
            raise ValueError("O parâmetro 'api_key' é obrigatório para o método 'openai'.")

        self.client = OpenAI(api_key=api_key)
        self.model_name = self.model_name or DEFAULT_OPENAI_MODEL
        logger.info(f"☁️ LLM OpenAI pronto: {self.model_name}")

    def _setup_local_model(self) -> None:
        """
        Orquestra o carregamento do modelo local, decidindo estratégias de quantização
        e correções de compatibilidade dinamicamente.

        Raises:
            RuntimeError: Se ocorrer qualquer erro durante o carregamento do Tokenizer ou do Modelo (ex: falta de VRAM, conexão).
        """
        self.model_name = self.model_name or DEFAULT_LOCAL_MODEL
        logger.info(f"🖥️ Preparando LLM Local '{self.model_name}' em: {self.device.upper()}")

        if self.device == "cpu":
            logger.warning("⚠️ ALERTA DE PERFORMANCE: Rodar LLM na CPU será significativamente lento.")

        try:
            # 1. Carrega Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._fix_tokenizer_padding()

            # 2. Constrói os argumentos de carregamento (Factory Method)
            loader_kwargs = self._build_loader_kwargs()

            # 3. Carrega o Modelo
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **loader_kwargs  # Desempacota os argumentos dinâmicos
            )

            logger.info("✅ Modelo local carregado com sucesso.")

        except Exception as e:
            logger.error(f"❌ Erro fatal ao carregar modelo: {e}")
            raise RuntimeError(f"Falha crítica ao carregar modelo local: {e}") from e

    # --- Métodos Auxiliares de Configuração (Modularização) ---

    def _build_loader_kwargs(self) -> Dict[str, Any]:
        """
        Constrói o dicionário de configurações (kwargs) para o `from_pretrained`.

        Isola a lógica de decisão:
        - Se for Unsloth/BNB: Não passa config de quantização (usa a nativa).
        - Se for modelo cru: Cria config BitsAndBytes.
        - Se for Phi-3/Falcon: Ativa trust_remote_code.

        Returns:
            Dict[str, Any]: Um dicionário contendo parâmetros como `device_map`, `quantization_config`, etc.
        """
        kwargs = {}

        # 1. Configuração por Tipo de Hardware
        if self.device == "cuda":
            # --- CORREÇÃO CRÍTICA PARA GPU DE 8GB ---
            # Força o modelo inteiro na GPU 0.
            # "auto" pode tentar jogar pedaços para a CPU, o que quebra modelos BNB 4-bit.
            kwargs["device_map"] = {"": 0}

            # Lógica de Quantização (Exclusiva de GPU)
            if self._is_pre_quantized():
                # Caso 1: Unsloth/BNB (Já vem pronto)
                logger.info("⚡ Modelo pré-quantizado detectado. Usando config nativa.")
            else:
                # Caso 2: Modelo Cru (Precisa comprimir agora)
                logger.info("🔧 Aplicando quantização 4-bit on-the-fly...")
                kwargs["quantization_config"] = self._get_bnb_config()

        else:
            # Fallback para CPU (Lento, mas funcional para testes sem quantização BNB)
            kwargs["device_map"] = "auto"

        # 2. Configuração Específica do Modelo (Remote Code / Atenção)
        if self._needs_remote_code():
            logger.warning(f"🛡️ Ativando 'trust_remote_code' para {self.model_name}")
            kwargs["trust_remote_code"] = True

            # Correção específica para Phi-3 e transformers novos
            if "Phi-3" in self.model_name:
                kwargs["attn_implementation"] = "eager"

        return kwargs

    def _needs_remote_code(self) -> bool:
        """
        Verifica se o modelo requer execução de código remoto (ex: arquiteturas novas).

        Returns:
            bool: True se o modelo estiver na lista de arquiteturas que exigem `trust_remote_code`.
        """
        keywords = ["Phi-3", "falcon", "mpt", "glm"]
        return any(k in self.model_name for k in keywords)

    def _is_pre_quantized(self) -> bool:
        """
        Detecta se o modelo já possui pesos quantizados.

        Returns:
            bool: True se o nome do modelo indicar quantização (bnb-4bit, awq, gptq).
        """
        # Unsloth usa 'bnb-4bit', outros usam 'awq', 'gptq'
        indicators = ["bnb-4bit", "awq", "gptq", "-quantized"]
        return any(i in self.model_name.lower() for i in indicators)

    def _get_bnb_config(self) -> BitsAndBytesConfig:
        """
        Gera a configuração padrão de quantização 4-bit (NF4).

        Returns:
            BitsAndBytesConfig: Objeto de configuração para injeção no `from_pretrained`.
        """
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    def _fix_tokenizer_padding(self) -> None:
        """Garante que o tokenizer tenha um token de pad (evita erros de geração)."""
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    # --- Geração de Texto ---

    def generate_response(self, prompt: str, system_prompt: str, max_new_tokens: int = 512,
                          temperature: float = 0.1, use_cache: bool = True) -> str:
        """
        Gera uma resposta textual. Atua como Dispatcher para Local ou OpenAI.

        Args:
            prompt (str): Entrada do usuário.
            system_prompt (str): Instruções do sistema.
            max_new_tokens (int): Limite de tokens a gerar.
            temperature (float): Criatividade (0.0 a 1.0).
            use_cache (bool): Ativa o cache KV para velocidade (True) ou desativa para economia de VRAM (False).

        Returns:
            str: O texto gerado pelo modelo, limpo e sem tokens especiais.

        Raises:
            LLMGenerationError: Wrapper para qualquer exceção que ocorra durante a inferência (rede ou local).
        """
        try:
            if self.method == 'openai':
                return self._generate_openai(prompt, system_prompt, max_new_tokens, temperature)
            elif self.method == 'local':
                return self._generate_local(prompt, system_prompt, max_new_tokens, temperature, use_cache)
        except Exception as e:
            logger.error(f"Erro na geração ({self.method}): {str(e)}")
            raise LLMGenerationError(f"Falha na geração de texto: {str(e)}") from e

    def _generate_openai(self, prompt: str, system_prompt: str, max_tokens: int, temperature: float) -> str:
        """
        Executa a geração via API da OpenAI.

        Returns:
            str: Conteúdo da resposta da API.
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
        Executa a inferência local com tratamento de erros de compatibilidade de cache.

        Args:
            prompt (str): Prompt do usuário.
            system_prompt (str): Prompt do sistema.
            max_new_tokens (int): Tokens máximos de saída.
            temperature (float): Temperatura de amostragem.
            use_cache (bool): Se True, usa cache KV.

        Returns:
            str: Texto decodificado.

        Raises:
            ValueError: Se o tamanho total (entrada + saída) exceder a janela de contexto.
            AttributeError/RuntimeError: Se houver falha na biblioteca transformers (relançado após tentativas).
        """
        # 1. Formata o prompt
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

        # 2. Verificação de Janela
        if input_len + max_new_tokens > self.context_window:
            msg = f"Prompt muito longo ({input_len}). Limite: {self.context_window}"
            logger.error(msg)
            raise ValueError(msg)

        # 3. Geração com Retry (Fallback)
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
            # Captura erro de compatibilidade 'seen_tokens' (comum no Phi-3 + Transformers novos)
            if "seen_tokens" in str(e) and use_cache:
                logger.warning("⚠️ Erro de Cache detectado. Tentando fallback com use_cache=False...")
                with torch.no_grad():
                    outputs = self.model.generate(
                        prompt_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=True if temperature > 0 else False,
                        temperature=temperature,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=False  # Desativa cache no retry
                    )
            else:
                raise e  # Repassa se for outro erro

        # 4. Decodificação
        generated_ids = outputs[0][input_len:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()