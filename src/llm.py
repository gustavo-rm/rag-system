from openai import OpenAI
import torch
from transformers import pipeline

class LLM:
    def __init__(self, method='openai', openai_api_key=None, local_model_name="EleutherAI/gpt-neo-2.7B"):
        """
        Inicializa a classe LLM com base no método desejado para gerar respostas.

        Parâmetros:
        - method: Define qual método será usado, 'openai' para a API da OpenAI ou 'local' para um modelo local do Hugging Face.
        - openai_api_key: Chave da API da OpenAI, necessária se o método escolhido for 'openai'.
        - local_model_name: Nome do modelo local a ser usado (disponível no Hugging Face), necessário se o método escolhido for 'local'.
        """
        self.method = method
        if method == 'openai':
            if openai_api_key is None:
                raise ValueError("Chave da API da OpenAI é necessária para usar OpenAI LLM.")
            self.client = OpenAI(api_key=openai_api_key)
        elif method == 'local':
            # Detecta se uma GPU está disponível e escolhe o dispositivo adequado (GPU ou CPU).
            device = 0 if torch.cuda.is_available() else -1  # 0 para GPU, -1 para CPU

            # Inicializa o pipeline de geração de texto do Hugging Face
            self.generator = pipeline(
                "text-generation",
                model=local_model_name,
                pad_token_id=50256,  # Especifica o token de preenchimento adequado ao modelo
                truncation=True,  # Trunca a entrada longa automaticamente
                device=device,  # Define GPU se disponível
                torch_dtype=torch.float16  # Usa half-precision (FP16) para reduzir o uso de memória
            )
        else:
            raise ValueError("Método de LLM inválido.")

    def generate_response(self, prompt, max_new_tokens=100, temperature=0.7):
        """
        Gera uma resposta para o prompt dado, usando o método definido (OpenAI ou local).

        Parâmetros:
        - prompt: String com o texto que será enviado ao modelo para gerar a resposta.
        - max_new_tokens: Número máximo de tokens a serem gerados na resposta (apenas para o modelo local).
        - temperature: Define a aleatoriedade da geração de texto (valores mais baixos produzem respostas mais determinísticas).

        Retorna:
        - A resposta gerada pelo modelo em forma de string.
        """
        if self.method == 'openai':
            # Faz uma requisição à API da OpenAI e obtém a resposta
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            # Extrai e retorna o conteúdo da resposta
            answer = response['choices'][0]['message']['content'].strip()
            return answer

        elif self.method == 'local':
            # Gera uma resposta usando o modelo local
            response = self.generator(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                truncation=True  # Trunca a resposta se ela exceder o número máximo de tokens
            )

            # Verifica o conteúdo da resposta gerada pelo modelo local
            print(f"Resposta do modelo local: {response}")

            # Extrai e retorna o texto gerado corretamente
            answer = response[0]['generated_text'].strip() if 'generated_text' in response[0] else response[0]['text'].strip()
            return answer
