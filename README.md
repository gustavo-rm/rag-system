# 🧠 RAGSystem - Retrieval-Augmented Generation System

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/models)
[![RAGAs](https://img.shields.io/badge/Evaluated%20by-RAGAs-orange)](https://github.com/explodinggradients/ragas)

**RAGSystem** é um sistema modular de Geração Aumentada por Recuperação (RAG). Ele processa documentos PDF, transforma seu conteúdo em uma base de conhecimento vetorial e permite que os usuários façam perguntas em linguagem natural, recebendo respostas baseadas exclusivamente nas fontes fornecidas.

Este projeto foi reestruturado para seguir as melhores práticas de design de software, oferecendo uma arquitetura flexível, componentes otimizados e uma avaliação do pipeline.

## 🔥 Funcionalidades Aprimoradas

-   **Extração e Limpeza de Conteúdo PDF**: Utiliza `PyMuPDF` para uma extração de texto de alta fidelidade, com um pipeline de limpeza que remove artefatos e normaliza o conteúdo para melhorar a qualidade.
-   **Chunking Semântico e Recursivo**: Implementa uma estratégia de divisão de texto avançada que preserva a coesão semântica e utiliza sobreposição (`overlap`) para não perder contexto entre os `chunks`.
-   **Geração de Embeddings Otimizada**: Suporte para modelos de embedding atualizados, com foco em modelos multilíngues (`sentence-transformers`) para alta performance em português e outros idiomas.
-   **Armazenamento Vetorial Híbrido**: Arquitetura flexível que suporta tanto bancos de dados de vetores locais e de código aberto (**ChromaDB**) quanto serviços gerenciados na nuvem (**Pinecone**), configuráveis através de um único ponto de entrada.
-   **Geração de Respostas com LLMs**: Suporte para LLMs via API (`OpenAI GPT-4o-mini`, etc.) e modelos locais otimizados para instruções (`Phi-3`, `Mistral`, `Llama 3`), utilizando templates de prompt aprimorados para reduzir alucinações.
-   **Avaliação do Pipeline**: Integração com o framework **RAGAs** para avaliar a qualidade do sistema em métricas cruciais como `faithfulness` (fidelidade), `answer_relevancy` (relevância da resposta) e `context_precision` (precisão do contexto).

## 📂 Estrutura do Projeto

A arquitetura foi refatorada para ser modular e escalável, utilizando pacotes para organizar responsabilidades.

```bash
├── src/
│   ├── stores/                # Pacote para armazenamento vetorial
│   │   ├── __init__.py        # Exporta a fábrica e as classes
│   │   ├── base.py            # Classe abstrata VectorStore (interface)
│   │   ├── chroma_store.py    # Implementação para ChromaDB (local)
│   │   ├── pinecone_store.py  # Implementação para Pinecone (nuvem)
│   │   └── factory.py         # Fábrica para selecionar o backend
│   │
│   ├── chunker.py             # Módulo de chunking avançado
│   ├── embedder.py            # Geração de embeddings otimizada
│   ├── evaluator.py           # Avaliador completo com RAGAs
│   ├── llm.py                 # Geração de texto com LLMs
│   ├── pdf_processor.py       # Extração e limpeza de PDFs
│   ├── pipeline.py          # Orquestrador principal do pipeline
│   └── utils.py               # (Funções utilitárias)
│
├── main.py                    # Script principal para executar o sistema
├── requirements.txt           # Dependências do projeto
├── .env.example               # Exemplo de arquivo para chaves de API
└── README.md                  # Documentação do projeto
````

## 🛠️ Tecnologias Utilizadas

  - **Linguagem**: Python 3.8+
  - **Frameworks**:
      - [PyTorch](https://pytorch.org/) para modelos de deep learning
      - [Hugging Face Transformers](https://huggingface.co/transformers/) para acesso a modelos locais
  - **Armazenamento Vetorial**:
      - [ChromaDB](https://www.trychroma.com/) (Local, Padrão)
      - [Pinecone](https://www.pinecone.io/) (Nuvem, Opcional)
  - **Avaliação**:
      - [RAGAs](https://github.com/explodinggradients/ragas) para métricas de RAG
      - `nltk` & `rouge-score` para métricas clássicas
  - **Bibliotecas Principais**:
      - `PyMuPDF` para manipulação de PDFs
      - `sentence-transformers` para embeddings
      - `openai` para a API da OpenAI
      - `accelerate` para carregamento otimizado de modelos
      - `python-dotenv` para gerenciamento de chaves

## 🚀 Como Executar o Projeto

### 1\. Pré-requisitos

Certifique-se de ter o Python 3.8+ e o `pip` instalado.

### 2\. Clone o Repositório

```bash
git clone [https://github.com/gustavo-rm/rag-system.git](https://github.com/gustavo-rm/rag-system)
cd seu-repositorio
```

### 3\. Instale as Dependências

Instale todas as dependências do `requirements.txt`.

```bash
pip install -r requirements.txt
```

**⚠️ Importante para Usuários de GPU (NVIDIA):** Para obter a máxima performance com modelos locais, é altamente recomendado instalar o [PyTorch com suporte a CUDA manualmente](https://pytorch.org/get-started/locally/) antes de rodar o comando acima.

### 4\. Configure as Variáveis de Ambiente

Crie um arquivo `.env` a partir do exemplo `.env.example`. As chaves são **opcionais** se você planeja usar apenas modelos e armazenamento locais.

```bash
# .env
OPENAI_API_KEY="sk-..."          # Necessário para usar modelos da OpenAI
PINECONE_API_KEY="sua-chave"     # Necessário para usar o backend do Pinecone
PINECONE_ENVIRONMENT="sua-regiao" # Necessário para usar o backend do Pinecone
```

### 5\. Execute o Sistema

A configuração do pipeline (qual LLM usar, qual vector store, etc.) é feita diretamente no `main.py`.

```bash
python main.py
```

## ⚙️ Configuração Flexível

O sistema utiliza um padrão de fábrica para selecionar os componentes, tornando a configuração fácil e centralizada no `main.py`.

**Exemplo 1: Configuração 100% Local (Padrão)**

```python
# Em main.py
config_store = {
    'type': 'chroma',
    'path': './db/meu_projeto_rag'
}
vector_store = get_vector_store(config_store)
llm = LLM(method='local', model_name='microsoft/Phi-3-mini-4k-instruct')
```

**Exemplo 2: Configuração Híbrida com OpenAI e Pinecone**

```python
# Em main.py
config_store = {
    'type': 'pinecone',
    'api_key': os.getenv("PINECONE_API_KEY"),
    'environment': os.getenv("PINECONE_ENVIRONMENT"),
    'index_name': 'meu-indice',
    'dimension': 768 # Dimensão do seu modelo de embedding
}
vector_store = get_vector_store(config_store)
llm = LLM(method='openai', api_key=os.getenv("OPENAI_API_KEY"))
```

## 🧪 Testes

A arquitetura modular e a classe `ComprehensiveEvaluator` facilitam a criação de testes de unidade para cada componente e testes de integração para o pipeline completo. A estrutura para testes futuros está em desenvolvimento.

## 🤝 Contribuições

Contribuições são bem-vindas\! Sinta-se à vontade para abrir issues e pull requests.

## 🛡️ Licença

Este projeto está licenciado sob a [Licença MIT](https://opensource.org/licenses/MIT).

```
```