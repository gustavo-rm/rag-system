import torch
import logging
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

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
DEFAULT_LOCAL_MODEL = "microsoft/Phi-3-mini-4k-instruct"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_CONTEXT_WINDOW = 4000


class LLM:
    """
    Interface unificada e otimizada para geração de texto usando LLMs Locais (Hugging Face) ou API (OpenAI).

    Esta classe abstrai a complexidade de carregar modelos, gerenciar quantização (4-bit),
    formatar templates de chat e validar janelas de contexto.

    Atributos:
        method (str): O método de execução ('local' ou 'openai').
        model_name (str): O identificador do modelo em uso.
        context_window (int): O limite máximo de tokens (entrada + saída) suportado.
        device (str): Dispositivo de execução ('cuda' ou 'cpu') - Apenas modo local.
    """

    def __init__(self, method: str = 'local', model_name: Optional[str] = None,
                 api_key: Optional[str] = None, context_window: int = DEFAULT_CONTEXT_WINDOW):
        """
        Inicializa a instância do LLM.

        Args:
            method (str): Escolha entre 'local' (execução na máquina) ou 'openai' (API).
                          Padrão: 'local'.
            model_name (str, opcional): ID do modelo no Hugging Face (ex: 'microsoft/Phi-3...')
                                        ou nome do modelo OpenAI (ex: 'gpt-4o').
                                        Se None, usa os padrões definidos em DEFAULT_*.
            api_key (str, opcional): Chave de API necessária se method='openai'.
            context_window (int): Limite de tokens para evitar truncamento silencioso.
                                  Padrão: 4000.

        Raises:
            ImportError: Se method='openai' e a biblioteca `openai` não estiver instalada.
            ValueError: Se method='openai' e `api_key` não for fornecida.
            ValueError: Se `method` não for 'local' nem 'openai'.
            RuntimeError: Se houver falha ao carregar o modelo local (ex: falta de VRAM, modelo inexistente).
        """
        self.method = method
        self.model_name = model_name
        self.context_window = context_window

        if method == 'openai':
            if not OpenAI:
                raise ImportError("Biblioteca 'openai' ausente. Instale com: pip install openai")
            if not api_key:
                raise ValueError("O parâmetro 'api_key' é obrigatório para o método 'openai'.")

            self.client = OpenAI(api_key=api_key)
            self.model_name = model_name or DEFAULT_OPENAI_MODEL
            logger.info(f"☁️ LLM OpenAI pronto: {self.model_name}")

        elif method == 'local':
            self.model_name = model_name or DEFAULT_LOCAL_MODEL
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(f"🖥️ Carregando LLM Local '{self.model_name}' em: {self.device.upper()}")

            if self.device == "cpu":
                logger.warning("⚠️ ALERTA DE PERFORMANCE: Rodar LLM na CPU será significativamente lento.")

            # Configuração de Quantização 4-bit (Apenas GPU NVIDIA)
            bnb_config = None
            if self.device == "cuda":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

                # Aviso de Segurança para trust_remote_code
                if "Phi-3" in self.model_name or "falcon" in self.model_name:
                    logger.warning("⚠️ ATENÇÃO: 'trust_remote_code=True' ativado. Use apenas modelos confiáveis.")

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    attn_implementation = "eager"
                )

                # Garante pad_token para modelos que não definem (evita erros na geração)
                if self.tokenizer.pad_token_id is None:
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            except Exception as e:
                raise RuntimeError(f"Falha crítica ao carregar modelo local: {e}")
        else:
            raise ValueError(f"Método '{method}' inválido. Escolha 'local' ou 'openai'.")

    def generate_response(self, prompt: str, system_prompt: str, max_new_tokens: int = 512,
                          temperature: float = 0.1) -> str:
        """
        Gera uma resposta textual baseada nos prompts fornecidos.

        Atua como um despachante (dispatcher), chamando a implementação específica
        (local ou remota) e padronizando o tratamento de erros.

        Args:
            prompt (str): A entrada do usuário (pergunta ou instrução).
            system_prompt (str): Instruções de comportamento para o modelo (persona, regras).
            max_new_tokens (int): O número máximo de tokens a serem gerados na resposta.
            temperature (float): Controle de criatividade (0.0 a 1.0).
                                 Valores baixos (0.1) são recomendados para tarefas factuais (RAG).

        Returns:
            str: O texto gerado pelo modelo, limpo e sem o prompt original.

        Raises:
            LLMGenerationError: Se ocorrer qualquer erro durante a chamada da API ou inferência local.
        """
        try:
            if self.method == 'openai':
                return self._generate_openai(prompt, system_prompt, max_new_tokens, temperature)
            elif self.method == 'local':
                return self._generate_local(prompt, system_prompt, max_new_tokens, temperature)
        except Exception as e:
            logger.error(f"Erro na geração ({self.method}): {str(e)}")
            # Encapsula o erro original em uma exceção de domínio do sistema
            raise LLMGenerationError(f"Falha na geração de texto: {str(e)}") from e

    def _generate_openai(self, prompt: str, system_prompt: str, max_tokens: int, temperature: float) -> str:
        """
        (Método Interno) Executa a geração via API da OpenAI.
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

    def _generate_local(self, prompt: str, system_prompt: str, max_new_tokens: int, temperature: float) -> str:
        """
        (Método Interno) Executa a inferência local usando Hugging Face Transformers.

        Realiza validação de tokens, gerenciamento de tensores e limpeza de memória.

        Raises:
            ValueError: Se o tamanho do prompt somado a `max_new_tokens` exceder `self.context_window`.
        """
        # 1. Aplica o template de chat específico do modelo
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        # Gera a string formatada (sem tokenizar ainda) para depuração ou log se necessário
        prompt_str = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Tokeniza e envia para a GPU
        inputs = self.tokenizer(prompt_str, return_tensors="pt").to(self.device)
        input_len = inputs.input_ids.shape[1]

        # 2. Verificação de Segurança da Janela de Contexto
        if input_len + max_new_tokens > self.context_window:
            msg = (f"Prompt muito longo ({input_len} tokens). Com a resposta ({max_new_tokens}), "
                   f"excederia o limite do modelo ({self.context_window}).")
            logger.error(msg)
            raise ValueError(msg)

        # 3. Geração Otimizada (Inferência Pura)
        # torch.no_grad() desativa o cálculo de gradientes, economizando VRAM e acelerando o processo.
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True if temperature > 0 else False,  # Greedy search se temp=0
                temperature=temperature if temperature > 0 else None,
                top_p=0.9,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=False
            )

        # 4. Decodificação e Fatiamento
        # O modelo retorna [prompt + resposta]. Cortamos a parte do prompt (input_len)
        generated_ids = outputs[0][input_len:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return response.strip()
