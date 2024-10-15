from openai import OpenAI
import torch
from transformers import pipeline

class LLM:
    def __init__(self, method='openai', openai_api_key=None, local_model_name="EleutherAI/gpt-neo-2.7B"):
        """
        method: 'openai' ou 'local'
        openai_api_key: chave da API da OpenAI, necessária se method='openai'
        local_model_name: nome do modelo local no Hugging Face, necessário se method='local'
        """
        self.method = method
        if method == 'openai':
            if openai_api_key is None:
                raise ValueError("Chave da API da OpenAI é necessária para usar OpenAI LLM.")
            self.client = OpenAI(api_key=openai_api_key)
        elif method == 'local':
            # Detectar se a GPU está disponível e escolher o dispositivo apropriado
            device = 0 if torch.cuda.is_available() else -1  # 0 é GPU, -1 é CPU

            # Inicializa o pipeline do Hugging Face para geração de texto
            self.generator = pipeline(
                "text-generation",
                model=local_model_name,
                pad_token_id=50256,  # Especificar token de preenchimento (depende do modelo)
                truncation=True,  # Truncar se a entrada for longa
                device=device,  # Usar GPU se disponível
                torch_dtype=torch.float16  # Half-precision para reduzir o uso de memória
            )
        else:
            raise ValueError("Método de LLM inválido.")

    def generate_response(self, prompt, max_tokens=300, temperature=0.7):
        if self.method == 'openai':
            # Fazer a requisição à API OpenAI
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            # Acessar o conteúdo correto da resposta
            answer = response['choices'][0]['message']['content'].strip()
            return answer

        elif self.method == 'local':
            # Gerar resposta com o modelo local
            response = self.generator(
                prompt,
                max_length=max_tokens,
                do_sample=True,
                temperature=temperature,
                truncation=True  # Truncar a resposta se ela exceder o tamanho máximo
            )

            # Verificar o conteúdo da resposta
            print(f"Response from local model: {response}")

            # Acessar o texto gerado corretamente
            answer = response[0]['generated_text'].strip() if 'generated_text' in response[0] else response[0]['text'].strip()
            return answer
