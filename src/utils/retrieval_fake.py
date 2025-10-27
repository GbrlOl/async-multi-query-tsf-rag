"""
Esta clase es auxiliar para mi sistema RAG
"""

from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from typing import List
from llama_index.core import VectorStoreIndex, QueryBundle
from llama_index.core.schema import NodeWithScore


class CustomFakeRetrieval(BaseRetriever):
    """
    Le puse Fake por emula el proceso de Retrieval, dado que ya se recuperaron los nodos, necesito los métodos y atributos de llama-index
    """

    def __init__(
        self,
        node_with_score: NodeWithScore,
        # reranker: SentenceTransformerRerank
    ) -> None:
        """Init params."""

        self._node_with_score = node_with_score
        # self._vector_retriever = vector_retriever
        # self.bm25_retriever = bm25_retriever
        # self.reranker = reranker

        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        # vector_nodes = self._vector_retriever.retrieve(query_bundle)
        # bm25_nodes = self.bm25_retriever.retrieve(query_bundle)

        # vector_nodes.extend(bm25_nodes)

        # retrieved_nodes = self.reranker.postprocess_nodes(
        # vector_nodes, query_bundle
        # )

        # return retrieved_nodes
        return self._node_with_score
