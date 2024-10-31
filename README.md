# 🧠 RAGSystem - Retrieval-Augmented Generation System

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10.0-red.svg)](https://pytorch.org/)

RAGSystem (Retrieval-Augmented Generation System) é um sistema de recuperação de informações combinado com geração de texto baseado em IA, capaz de realizar consultas sobre documentos PDF e gerar respostas baseadas no contexto extraído.

## 🔥 Funcionalidades

- **Extração de conteúdo PDF**: Processa e extrai texto de arquivos PDF.
- **Chunking do conteúdo**: Divide o texto em segmentos (chunks) para facilitar a indexação.
- **Geração de Embeddings**: Suporta a geração de embeddings com `OpenAI` ou `Sentence-BERT` para indexação e consulta.
- **Armazenamento de Embeddings**: Integração com Pinecone para armazenamento e busca vetorial.
- **Geração de Texto**: Suporte para modelos de linguagem (LLM), utilizando tanto OpenAI GPT quanto modelos locais.

## 📂 Estrutura do Projeto

```bash
├── src/
│   ├── chunker.py         # Função para dividir o texto extraído em chunks
│   ├── embedder.py        # Classe para geração de embeddings
│   ├── embedding_store.py # Armazenamento e busca de embeddings usando Pinecone
│   ├── evaluator.py       # Avaliação de resultados com métricas (ex: ROUGE)
│   ├── llm.py             # Geração de texto com LLMs (OpenAI e modelos locais)
│   ├── pdf_extractor.py   # Extração de texto de PDFs
│   ├── rag_system.py      # Sistema principal do RAG que integra todos os componentes
│   └── utils.py           # Funções utilitárias
├── main.py                # Script principal para executar o sistema
├── requirements.txt       # Dependências do projeto
└── README.md              # Documentação do projeto
```

## 🛠️ Tecnologias Utilizadas

- **Linguagem**: Python 3.8+
- **Frameworks**: 
  - [PyTorch](https://pytorch.org/) para modelos locais de LLM
  - [Hugging Face Transformers](https://huggingface.co/transformers/) para integração com modelos de texto
- **APIs**:
  - [OpenAI GPT](https://beta.openai.com/)
  - [Pinecone](https://www.pinecone.io/) para armazenamento de vetores
- **Bibliotecas**:
  - `PyPDF2` e `pymupdf` para manipulação de PDFs
  - `nltk` para tokenização
  - `sentence-transformers` para embeddings
  - `dotenv` para carregar variáveis de ambiente

## 🚀 Como Executar o Projeto

### 1. Pré-requisitos

Certifique-se de ter o Python 3.8+ e o `pip` instalado. 

```bash
python --version
pip --version
```

### 2. Clone o Repositório

Clone o repositório do projeto para sua máquina local:

```bash
git clone https://github.com/gustavo-rm/rag-system.git
cd RAGSystem
```

### 3. Instale as Dependências

Instale todas as dependências necessárias listadas no arquivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Configure as Chaves de API

Crie um arquivo `.env` na raiz do projeto e insira suas chaves da OpenAI e Pinecone:

```bash
# .env
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENVIRONMENT=your-pinecone-environment
OPENAI_API_KEY=your-openai-api-key
```

### 5. Execute o Sistema

Agora você pode rodar o sistema usando o `main.py`:

```bash
python main.py
```

## 📝 Como Usar

1. Insira o caminho para o PDF que você deseja consultar.
2. O sistema irá dividir o texto em chunks, gerar embeddings e armazená-los no Pinecone.
3. Você pode fazer perguntas, e o sistema irá gerar uma resposta com base no conteúdo do PDF.

## ⚙️ Configurações

- **Métodos de Embeddings**: 
  - `"openai"` (Usa embeddings da OpenAI)
  - `"sbert"` (Usa Sentence-BERT para embeddings locais)
  
- **LLM Methods**: 
  - `"openai"` (Usa a API OpenAI GPT)
  - `"local"` (Usa um modelo local como GPT-Neo)

As configurações de chunking, embeddings e LLM podem ser ajustadas diretamente no código no momento de inicialização do sistema.

## 🧪 Testes

Ainda não configurado.

## 🐛 Problemas Conhecidos

- **Memória Insuficiente na GPU**: Se você estiver rodando em um ambiente com GPU limitada, utilize modelos menores ou rode na CPU.
- **Limitações de Token**: OpenAI tem limites de token em suas requisições, fique atento ao tamanho das consultas.

## 🤝 Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues e pull requests.

1. Crie um fork do projeto
2. Crie um branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Comite suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Dê um push para o branch (`git push origin feature/AmazingFeature`)
5. Abra um pull request

## 🛡️ Licença

Este projeto está licenciado sob a [Licença MIT](https://opensource.org/licenses/MIT).
