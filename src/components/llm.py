import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
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
        self.method = method
        self.model_name = model_name

        if method == 'openai':
            # ... (código da OpenAI permanece o mesmo)
            if not api_key:
                raise ValueError("Chave da API é necessária para o método 'openai'.")
            self.client = OpenAI(api_key=api_key)
            if not self.model_name:
                self.model_name = "gpt-3.5-turbo"

        elif method == 'local':
            if not self.model_name:
                self.model_name = "microsoft/Phi-3-mini-4k-instruct"

            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Carregando modelo local '{self.model_name}' no dispositivo: {device}")

            # NOVO: Define a configuração de quantização em 4 bits
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4"
            )

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                # NOVO: Passa a configuração de quantização para o modelo
                quantization_config=quantization_config if device == "cuda" else None,
                device_map="auto"
            )
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
        # (Este método permanece o mesmo, sem alterações)
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
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            outputs = self.generator(
                full_prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature if temperature > 0 else None,
                top_p=0.95,
                eos_token_id=self.tokenizer.eos_token_id
            )

            result = outputs[0]['generated_text']
            answer = result[len(full_prompt):].strip()
            return answer