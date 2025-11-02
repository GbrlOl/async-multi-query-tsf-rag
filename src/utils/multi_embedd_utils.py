from llama_index.core import (
    StorageContext,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.nomic import NomicEmbedding
from llama_index.core import PromptTemplate
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine

from typing import Literal, List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import copy

from prompts import RAG_QA_TEMPLATE
from llms_modules import MultiQueryGeneration
from utils import CustomFakeRetrieval
from dotenv import load_dotenv
from tqdm import tqdm
import os

multi_query_generator = MultiQueryGeneration()

# Cargar el archivo .env
load_dotenv()

# Acceder a las variables de entorno
api_key = os.getenv("OPENAI_API_KEY")
nomic_api_key = os.getenv("NOMIC_API_KEY")


class EmbeddingConfig:
    """Configuración para un modelo de embedding específico"""

    def __init__(
        self,
        name: str,
        embedding_type: Literal["huggingface", "nomic"],
        model_name: str,
        path_vector_storage: str,
        **kwargs,
    ):
        self.name = name
        self.embedding_type = embedding_type
        self.model_name = model_name
        self.path_vector_storage = path_vector_storage
        self.kwargs = kwargs


class AsyncMultiEmbeddingRAGV2:
    """
    Sistema RAG que permite usar múltiples modelos de embeddings simultáneamente.
    Cada modelo de embedding puede tener su propio índice vectorial.
    """

    def __init__(
        self,
        embedding_configs: List[EmbeddingConfig],
        top_k: int = 13,
        cross_encoder_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n=5,
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ):
        """
        Inicializa el sistema RAG con múltiples embeddings.

        Args:
            embedding_configs: Lista de configuraciones de embeddings
            top_k: Número de documentos similares a recuperar por embedding
            cross_encoder_model_name: Modelo para reranking
            top_n: Número de documentos finales después del reranking
            llm_model: Modelo de LLM a utilizar
            temperature: Temperatura para el LLM
        """
        self.embedding_configs = embedding_configs
        self.top_k = top_k
        self.cross_encoder_model_name = cross_encoder_model_name
        self.top_n = top_n
        self.llm_model = llm_model
        self.temperature = temperature

        # Variables de entorno
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.nomic_api_key = os.getenv("NOMIC_API_KEY")

        # Diccionarios para almacenar componentes por embedding
        self.embed_models: Dict[str, Any] = {}
        self.vector_indices: Dict[str, Any] = {}
        self.retrievers: Dict[str, Any] = {}

        # Componentes compartidos
        self.cross_encoder = None

        # Configurar el sistema
        self._setup_llm()
        self._setup_embeddings_and_indices()
        self._setup_cross_encoder()

    def _create_embedding_model(self, config: EmbeddingConfig):
        """Crea un modelo de embedding específico."""
        if config.embedding_type == "huggingface":
            return HuggingFaceEmbedding(model_name=config.model_name, **config.kwargs)
        elif config.embedding_type == "nomic":
            if not self.nomic_api_key:
                raise ValueError(
                    "NOMIC_API_KEY no encontrada en las variables de entorno"
                )

            return NomicEmbedding(
                api_key=self.nomic_api_key,
                dimensionality=768,
                model_name=config.model_name,
                **config.kwargs,
            )
        else:
            raise ValueError("embedding_type debe ser 'huggingface' o 'nomic'")

    def _setup_embeddings_and_indices(self):
        """Configura todos los modelos de embeddings e índices."""
        print("Configurando modelos de embeddings e índices...")

        for config in self.embedding_configs:
            print(f"Configurando embedding: {config.name}")

            # Crear modelo de embedding
            embed_model = self._create_embedding_model(config)
            self.embed_models[config.name] = embed_model

            # Cargar índice vectorial específico para este embedding
            try:
                # Temporalmente establecer el embedding en Settings
                original_embed_model = getattr(Settings, "embed_model", None)
                Settings.embed_model = embed_model

                vector_store = FaissVectorStore.from_persist_dir(
                    config.path_vector_storage
                )
                storage_context = StorageContext.from_defaults(
                    vector_store=vector_store, persist_dir=config.path_vector_storage
                )

                vector_index = load_index_from_storage(storage_context=storage_context)
                self.vector_indices[config.name] = vector_index

                # Crear retriever
                retriever = vector_index.as_retriever(similarity_top_k=self.top_k)
                self.retrievers[config.name] = retriever

                # Restaurar el embedding original en Settings
                if original_embed_model is not None:
                    Settings.embed_model = original_embed_model

                print(f"✓ Embedding {config.name} configurado correctamente")

            except Exception as e:
                raise Exception(
                    f"Error al cargar el índice para {config.name}: {str(e)}"
                )

    def _setup_llm(self):
        """Configura el modelo de lenguaje."""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY no encontrada en las variables de entorno")

        Settings.llm = OpenAI(
            model=self.llm_model,
            temperature=self.temperature,
            api_key=self.openai_api_key,
        )

    def _setup_cross_encoder(self):
        """Carga el modelo de re-rank"""
        print("Cargando Cross-Encoder...")
        self.cross_encoder = SentenceTransformerRerank(
            model=self.cross_encoder_model_name,
            top_n=self.top_n,
            keep_retrieval_score=True,
        )

    def _retrieval_with_embedding(
        self, embedding_name: str, queries: List[str]
    ) -> List[List]:
        """Realiza retrieval para un embedding específico."""
        retriever = self.retrievers[embedding_name]
        embed_model = self.embed_models[embedding_name]

        # Temporalmente establecer el embedding específico
        original_embed_model = getattr(Settings, "embed_model", None)
        Settings.embed_model = embed_model

        try:
            with ThreadPoolExecutor() as executor:
                resultados = list(executor.map(retriever.retrieve, queries))

            return resultados
        finally:
            # Restaurar embedding original
            if original_embed_model is not None:
                Settings.embed_model = original_embed_model

    def multi_embedding_retrieval(self, queries: List[str]) -> Dict[str, List[List]]:
        """Realiza retrieval con todos los embeddings de manera paralela."""
        print("Iniciando retrieval con múltiples embeddings...")

        all_results = {}

        # Usar ThreadPoolExecutor para paralelizar entre embeddings
        with ThreadPoolExecutor() as executor:
            # Crear tareas para cada embedding
            future_to_embedding = {
                executor.submit(
                    self._retrieval_with_embedding, embedding_name, queries
                ): embedding_name
                for embedding_name in self.embed_models.keys()
            }

            # Recopilar resultados
            for future in tqdm(future_to_embedding, desc="Procesando embeddings"):
                embedding_name = future_to_embedding[future]
                try:
                    results = future.result()
                    all_results[embedding_name] = results
                    print(f"✓ Retrieval completado para embedding: {embedding_name}")
                except Exception as e:
                    print(f"✗ Error en embedding {embedding_name}: {str(e)}")
                    all_results[embedding_name] = []

        return all_results

    def aggregate_and_filter_nodes(
        self, all_embedding_results: Dict[str, List[List]]
    ) -> List:
        """Agrega resultados de todos los embeddings y filtra duplicados."""
        print("Agregando y filtrando nodos de todos los embeddings...")

        # Desempaquetar todos los nodos de todos los embeddings
        all_nodes = []

        for embedding_name, embedding_results in all_embedding_results.items():
            print(f"Procesando nodos de embedding: {embedding_name}")
            for query_results in embedding_results:
                all_nodes.extend(query_results)

        print(f"Total de nodos antes del filtrado: {len(all_nodes)}")

        # Filtrar nodos duplicados
        filtered_nodes = []
        seen = set()

        for node in all_nodes:
            # Extraer metadatos relevantes
            file_name = node.metadata.get("file_name", "")
            page_label = node.metadata.get("page_label", "")

            # Crear clave única
            key = (file_name, page_label)

            if key not in seen:
                seen.add(key)
                filtered_nodes.append(node)

        print(f"Total de nodos después del filtrado: {len(filtered_nodes)}")
        return filtered_nodes

    def rerank_with_multiple_queries(
        self, queries: List[str], nodes: List
    ) -> List[List]:
        """Aplica reranking usando múltiples queries."""

        def rerank_single_query(query: str):
            return self.cross_encoder.postprocess_nodes(nodes, query_str=query)

        with ThreadPoolExecutor() as executor:
            rerank_results = list(executor.map(rerank_single_query, queries))

        return rerank_results

    def query(self, query: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Ejecuta el pipeline completo con múltiples embeddings.

        Returns:
            Tuple con la respuesta final y metadata del proceso
        """
        # Etapa 1: Generar múltiples queries
        lista_querys_generadas = multi_query_generator.generation(query)
        print(f"Etapa 1 - Total de querys generadas: {len(lista_querys_generadas)}")

        # Etapa 2: Retrieval con múltiples embeddings
        print("Etapa 2 - Retrieval con múltiples embeddings")
        all_embedding_results = self.multi_embedding_retrieval(lista_querys_generadas)

        # Etapa 3: Agregar y filtrar nodos
        print("Etapa 3 - Agregación y filtrado de nodos")
        filtered_nodes = self.aggregate_and_filter_nodes(all_embedding_results)

        # Etapa 4: Reranking
        print("Etapa 4 - Reranking")
        rerank_results = self.rerank_with_multiple_queries(
            lista_querys_generadas, filtered_nodes
        )

        # Etapa 5: Generar respuesta final
        print("Etapa 5 - Generación de respuesta")
        retrieval_fake = CustomFakeRetrieval(filtered_nodes)
        query_engine = RetrieverQueryEngine.from_args(
            retrieval_fake, text_qa_template=PromptTemplate(RAG_QA_TEMPLATE)
        )

        response = query_engine.query(lista_querys_generadas[0])

        # Metadata del proceso
        metadata = {
            "queries_generated": lista_querys_generadas,
            "embedding_results": all_embedding_results,
            "total_nodes_before_filtering": sum(
                len(
                    [
                        node
                        for query_results in embedding_results
                        for node in query_results
                    ]
                )
                for embedding_results in all_embedding_results.values()
            ),
            "total_nodes_after_filtering": len(filtered_nodes),
            "rerank_results": rerank_results,
            "embeddings_used": list(self.embed_models.keys()),
        }

        return response, metadata

    def get_embedding_info(self) -> Dict[str, Dict]:
        """Retorna información sobre los embeddings configurados."""
        info = {}
        for config in self.embedding_configs:
            info[config.name] = {
                "type": config.embedding_type,
                "model_name": config.model_name,
                "storage_path": config.path_vector_storage,
                "has_index": config.name in self.vector_indices,
            }
        return info
