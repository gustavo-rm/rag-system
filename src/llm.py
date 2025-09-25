# /src/llm.py (Versão Refatorada)

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
from typing import Optional


# --- Recomendações de Modelos Locais (Instruction-Tuned) ---
# Modelos menores e mais rápidos, ótimos para começar:
# - "microsoft/Phi-3-mini-4k-instruct"
# Modelos maiores e mais capazes (requerem mais VRAM):
# - "mistralai/Mistral-7B-Instruct-v0.2"
# - "meta-llama/Meta-Llama-3-8B-Instruct"

class LLM:
    def __init__(self, method: str = 'local', model_name: Optional[str] = None, api_key: Optional[str] = None):
        """
        Inicializa a classe LLM.

        Parâmetros:
        - method (str): 'openai' ou 'local'.
        - model_name (str): Nome do modelo a ser usado.
        - api_key (str): Chave da API da OpenAI, necessária para method='openai'.
        """
        self.method = method
        self.model_name = model_name

        if method == 'openai':
            if not api_key:
                raise ValueError("Chave da API é necessária para o método 'openai'.")
            self.client = OpenAI(api_key=api_key)
            if not self.model_name:
                self.model_name = "gpt-3.5-turbo"  # Default para OpenAI

        elif method == 'local':
            if not self.model_name:
                self.model_name = "microsoft/Phi-3-mini-4k-instruct"  # Um default moderno e leve

            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Carregando modelo local '{self.model_name}' no dispositivo: {device}")

            # Usar AutoModel e AutoTokenizer para mais controle
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto"  # Deixa a biblioteca decidir a melhor alocação (CPU/GPU)
            )
            # O pipeline ainda é uma forma fácil de usar o modelo e o tokenizador juntos
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto"
            )
        else:
            raise ValueError("Método de LLM inválido. Escolha 'openai' ou 'local'.")

    def generate_response(self, prompt: str, system_prompt: str, max_new_tokens: int = 250,
                          temperature: float = 0.1) -> str:
        """
        Gera uma resposta para o prompt, considerando um prompt de sistema.
        """
        if self.method == 'openai':
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Erro ao chamar a API da OpenAI: {e}")
                return "Erro ao gerar resposta da OpenAI."

        elif self.method == 'local':
            # Formatação específica para modelos de chat que usam tokens especiais
            # (Ex: Phi-3, Llama 3, Mistral)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            # `apply_chat_template` formata o prompt da maneira que o modelo espera
            full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            outputs = self.generator(
                full_prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature if temperature > 0 else None,
                top_p=0.95,
                eos_token_id=self.tokenizer.eos_token_id
            )

            # CORREÇÃO CRÍTICA: Remove o prompt da saída para retornar apenas a resposta.
            result = outputs[0]['generated_text']
            # O `full_prompt` é o que foi enviado, a resposta é o que vem depois
            answer = result[len(full_prompt):].strip()
            return answer