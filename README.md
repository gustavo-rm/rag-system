---

# 🧠 RAGSystem v3.0 - Enterprise Retrieval-Augmented Generation

**RAGSystem v3.0** é uma plataforma avançada de Geração Aumentada por Recuperação (RAG), projetada para alta precisão, baixa latência e robustez. Diferente de pipelines RAG básicos, esta versão implementa estratégias de ponta como **Busca Híbrida (Keyword + Vetorial)**, **Re-ranking com Cross-Encoders**, **Roteamento Semântico de Perguntas** e **Caching Inteligente de Duas Camadas**.

O sistema foi arquitetado para ser agnóstico em relação ao modelo (suporta LLMs Locais via HuggingFace e GPT-4 via OpenAI) e ao banco de dados (ChromaDB ou Pinecone).

---

## 🔥 O Que Há de Novo na v3.0?

A versão 3.0 introduz um pipeline de processamento mais sofisticado:

### 1. Recuperação e Precisão (Retrieval & Precision)

* **Busca Híbrida (Hybrid Search):** Combina a busca semântica (Embeddings/Dense) com a busca por palavras-chave (BM25/Sparse). Isso resolve o problema de encontrar termos exatos, IDs ou nomes próprios que modelos vetoriais as vezes perdem.
* **Re-ranking (Cross-Encoder):** Uma segunda etapa de filtragem onde um modelo "Juiz" (`BAAI/bge-reranker`) reordena os documentos recuperados, garantindo que o contexto enviado ao LLM seja extremamente relevante.
* **Query Routing (Roteador Semântico):** Um classificador inteligente analisa a pergunta do usuário e escolhe a melhor estratégia:
* *MultiQuery:* Para perguntas vagas.
* *HyDE (Hypothetical Document Embeddings):* Para perguntas conceituais.
* *NoOp:* Para perguntas diretas e específicas.



### 2. Performance e Latência

* **Caching L1 & L2:**
* *L1 (Exato):* Retorno instantâneo para perguntas repetidas.
* *L2 (Semântico):* Usa FAISS para encontrar perguntas *similares* ("Quanto custa?" ≈ "Qual o preço?") e reaproveitar respostas.


* **GPU Acceleration:** Detecção automática de CUDA e uso de `Mixed Precision` para inferência e treinamento.

### 3. Qualidade de Dados

* **Query Correction:** Pipeline de pré-processamento que usa `LanguageTool` e `Regex` para corrigir erros gramaticais e de digitação do usuário antes da busca.
* **Embedding Fine-Tuning:** Módulo dedicado para treinar o modelo de embedding nos seus próprios dados (Domain Adaptation), melhorando a compreensão de jargões técnicos.

---

## 🧩 Arquitetura do Pipeline

```mermaid graph TD
    User[Usuário] --> Corrector[Query Corrector]
    Corrector --> Cache{Cache Check}
    Cache -- Hit --> Response[Resposta Imediata]
    Cache -- Miss --> Router[Query Router]
    
    Router -->|Conceitual| HyDE[HyDE Transformer]
    Router -->|Ambígua| Multi[MultiQuery Transformer]
    Router -->|Específica| NoOp[Direct Search]
    
    HyDE & Multi & NoOp --> Hybrid[Hybrid Retriever]
    Hybrid -->|BM25 + Vetor| Candidates[Candidatos]
    
    Candidates --> Reranker[Cross-Encoder ReRanker]
    Reranker -->|Top N Contexts| LLM[LLM Generator]
    
    LLM --> Response
```

## 📂 Estrutura do Projeto

O código foi modularizado para facilitar manutenção e escalabilidade.

```bash
├── data/                      # Persistência de dados (ChromaDB, PDFs e datasets de treino)
├── models/                    # Diretório para salvar modelos locais e finetuned
├── src/
│   ├── caching/               # Camada de Cache Inteligente
│   │   ├── cache_manager.py   # Cache L1 (Exato/Hash Map)
│   │   └── semantic_cache.py  # Cache L2 (Semântico/FAISS)
│   │
│   ├── chat/                  # Gestão de Conversação
│   │   ├── chatbot.py         # Controlador do fluxo de chat
│   │   └── history.py         # Histórico com janela deslizante e persistência
│   │
│   ├── components/            # Componentes Core de IA
│   │   ├── embedder.py        # Gerador de Embeddings (Suporte a Batching/GPU)
│   │   ├── hybrid_retriever.py# Busca Híbrida (BM25 + Vetor)
│   │   ├── llm.py             # Interface unificada para LLMs (Local/OpenAI)
│   │   └── reranker.py        # Cross-Encoder para reclassificação
│   │
│   ├── evaluation/            # Auditoria de Qualidade
│   │   └── evaluator.py       # Avaliação com RAGAS e Juiz LLM
│   │
│   ├── ingestion/             # Pipeline de Ingestão (ETL)
│   │   ├── chunker.py         # Divisão de texto (Semantic Chunking)
│   │   └── pdf_processor.py   # Extração e limpeza de PDFs
│   │
│   ├── preprocessing/         # Pré-processamento de Entrada
│   │   └── query_corrector.py # Correção ortográfica e normalização
│   │
│   ├── query_transformers/    # Estratégias de Expansão de Query
│   │   ├── base.py
│   │   ├── hyde_transformer.py
│   │   ├── multi_query_transformer.py
│   │   └── noop_transformer.py
│   │
│   ├── routing/               # Roteamento Semântico
│   │   └── query_router.py    # Classificador de intenção de busca
│   │
│   ├── stores/                # Adaptadores de Vector Store
│   │   ├── base.py
│   │   ├── chroma_store.py    # Backend Local
│   │   ├── factory.py         # Factory Pattern
│   │   └── pinecone_store.py  # Backend Nuvem
│   │
│   ├── training/              # Módulo de Fine-Tuning (Domain Adaptation)
│   │   ├── generators/        # Geração de dados de treino
│   │   │   ├── base.py
│   │   │   ├── file_generator.py
│   │   │   └── synthetic_generator.py # Geração sintética (GPL)
│   │   └── trainer.py         # Loop de treinamento
│   │
│   ├── utils/                 # Utilitários
│   │   └── logger.py          # Configuração de logging centralizado
│   │
│   └── pipeline.py            # Orquestrador do RAGSystem
│
├── main.py                    # Aplicação Principal (Chatbot CLI)
├── train_embedding.py         # Script CLI para Fine-Tuning de Embeddings
├── test_rag.py                # Script de testes e avaliação
├── requirements.txt           # Dependências do projeto
└── README.md                  # Documentação
```

---

## 🛠️ Stack Tecnológico

* **LLMs & NLP:** [LangChain](https://www.langchain.com/), [Hugging Face Transformers](https://huggingface.co/), [Sentence-Transformers](https://www.sbert.net/).
* **Vector Search:** [ChromaDB](https://www.trychroma.com/) (Local), [Pinecone](https://www.pinecone.io/) (Cloud), [FAISS](https://github.com/facebookresearch/faiss) (Cache).
* **Retrieval:** `Rank-BM25` (Sparse), `BAAI/bge-reranker` (Re-ranking).
* **Correction:** `LanguageTool` (Gramática), `Unidecode` (Normalização).
* **Evaluation:** [RAGAS](https://github.com/explodinggradients/ragas).

---

## 🚀 Como Executar

### 1. Instalação

```bash
# Clone o repositório
git clone https://github.com/gustavo-rm/rag-system.git
cd rag-system-v3

# Crie um ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instale as dependências
pip install -r requirements.txt
```

> **Nota:** Para usar modelos locais com GPU, instale o PyTorch com suporte a CUDA antes de rodar o `requirements.txt`.

### 2. Configuração (.env)

Copie o `.env.example` para `.env`:

```ini
# Opcional: Se usar modelos da OpenAI
OPENAI_API_KEY="sk-..."

# Opcional: Se usar Pinecone
PINECONE_API_KEY="..."
PINECONE_ENV="..."
```

### 3. Rodando o Chatbot

O sistema irá ingerir automaticamente o PDF configurado (se ainda não o fez) e iniciar o chat.

```bash
python main.py
```

### 4. Treinamento de Embeddings (Fine-Tuning)

A v3.0 permite que você refine o modelo de embeddings usando seus próprios PDFs para entender melhor o vocabulário do seu domínio.

```bash
# Gera 50 exemplos sintéticos usando o PDF e treina por 3 épocas
python train_embedding.py --mode synthetic --input data/pdfs/meu_manual.pdf --num_gen 50 --epochs 3
```

O novo modelo será salvo em `models/finetuned_v3` e pode ser configurado no `main.py`.

---

## 📊 Avaliação e Métricas

O sistema utiliza o framework **RAGAS** com um conceito de "Juiz LLM" (geralmente GPT-4) para avaliar a qualidade das respostas geradas pelo modelo local.

Métricas monitoradas:

1. **Faithfulness:** A resposta deriva puramente do contexto recuperado? (Anti-Alucinação).
2. **Answer Relevancy:** A resposta atende à pergunta do usuário?
3. **Context Precision:** O HybridRetriever trouxe os documentos certos no topo?

Para rodar a avaliação (requer Dataset Golden ou Ground Truth):

```python
from src.evaluation.evaluator import RAGEvaluator
evaluator = RAGEvaluator()
# ... (vide código de exemplo na pasta tests)
```

---

## 🤝 Contribuições

Contribuições são bem-vindas! Por favor, siga o fluxo:

1. Fork o projeto.
2. Crie uma branch (`git checkout -b feature/MinhaFeature`).
3. Commit suas mudanças (`git commit -m 'feat: Adiciona nova estratégia de cache'`).
4. Push para a branch (`git push origin feature/MinhaFeature`).
5. Abra um Pull Request.

## 📜 Licença

Distribuído sob a licença MIT. Veja `LICENSE` para mais informações.