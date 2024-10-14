from openai import OpenAI
from transformers import pipeline


class LLM:
    def __init__(self, method='openai', openai_api_key=None, local_model_name="EleutherAI/gpt-neo-2.7B"):
        """
        method: 'openai' ou 'local'
        openai_api_key: chave da API da OpenAI, necessária se method='openai'
        local_model_name: nome do modelo local no Hugging Face, necessário se method='local'
        """
        self.client = OpenAI(api_key=openai_api_key)
        self.method = method
        if method == 'openai':
            if openai_api_key is None:
                raise ValueError("Chave da API da OpenAI é necessária para usar OpenAI LLM.")
        elif method == 'local':
            self.generator = pipeline("text-generation", model=local_model_name)
        else:
            raise ValueError("Método de LLM inválido.")

    def generate_response(self, prompt, max_tokens=300, temperature=0.7):
        if self.method == 'openai':
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            answer = response.choices[0].text.strip()
            return answer
        elif self.method == 'local':
            response = self.generator(prompt, max_length=max_tokens, do_sample=True, temperature=temperature)
            answer = response[0].generated_text.strip()
            return answer
