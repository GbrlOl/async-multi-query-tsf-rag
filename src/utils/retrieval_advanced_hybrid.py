from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from typing import List
from llama_index.core import VectorStoreIndex, QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.retrievers.bm25 import BM25Retriever


class EmbeddingBM25RerankRetriever(BaseRetriever):
    """Este es un recuperador customizado, donde recupero los nodos del embeddings y del BM25, luego con extend los concateno.
    Esto me permite tener un único recuperador utilizando los objetos de LlamaIndex
    """

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        bm25_retriever: BM25Retriever,
        reranker: SentenceTransformerRerank,
    ) -> None:
        """Init params."""

        self._vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker

        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        bm25_nodes = self.bm25_retriever.retrieve(query_bundle)

        vector_nodes.extend(bm25_nodes)

        retrieved_nodes = self.reranker.postprocess_nodes(vector_nodes, query_bundle)

        return retrieved_nodes
        # return vector_nodes
