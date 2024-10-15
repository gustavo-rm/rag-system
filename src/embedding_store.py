import pinecone


class EmbeddingStore:
    def __init__(self, pinecone_api_key, pinecone_environment, dimension=384, index_name="my-vector-index"):
        """
        Inicializa o armazenamento de embeddings usando Pinecone.
        - pinecone_api_key: chave da API do Pinecone
        - pinecone_environment: ambiente Pinecone (ex: 'us-west1-gcp')
        - dimension: dimensão dos embeddings
        - index_name: nome do índice Pinecone a ser utilizado
        """
        # Inicializa o Pinecone usando a nova API
        self.pinecone = pinecone.Pinecone(api_key=pinecone_api_key)

        self.index_name = index_name
        self.dimension = dimension

        # Verifica se o índice já existe, senão cria um novo
        if self.index_name not in self.pinecone.list_indexes().names():
            self.pinecone.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric='euclidean',  # Escolha a métrica que você preferir
                spec=pinecone.ServerlessSpec(cloud='aws', region=pinecone_environment)
            )

        # Conecta ao índice
        self.index = self.pinecone.Index(self.index_name)

    def store_embeddings(self, embeddings, ids=None):
        """
        Armazena embeddings no índice do Pinecone.
        - embeddings: lista de embeddings a serem armazenados.
        - ids: lista de IDs associada aos embeddings.
        """
        if ids is None:
            ids = [str(i) for i in range(len(embeddings))]

        # Insere os embeddings no índice
        vectors = list(zip(ids, embeddings))
        self.index.upsert(vectors)

    def search(self, query_embedding, top_k=5, namespace=None):
        """
        Busca embeddings mais próximos no Pinecone.
        - query_embedding: embedding da consulta.
        - top_k: número de resultados a serem retornados.
        - namespace: opcional, para organizar a busca em um namespace específico.
        """
        # Converter embedding para lista de floats
        query_embedding = [float(x) for x in query_embedding]

        # Verificar o formato do embedding
        print(f"Embedding para busca: {query_embedding[:10]}...")

        # Realizar a busca no Pinecone usando 'vector' e 'namespace' (opcional)
        result = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=namespace,
            include_values=True  # Incluir os valores dos vetores correspondentes no resultado
        )

        # Verificar o resultado
        if 'matches' in result:
            return result['matches']
        else:
            print(f"Erro na busca no Pinecone: {result}")
            return []

    def delete_index(self):
        """
        Deleta o índice atual no Pinecone.
        Útil para limpar dados ou redefinir o índice.
        """
        self.pinecone.delete_index(self.index_name)
