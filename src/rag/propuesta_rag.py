from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    ServiceContext,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.nomic import NomicEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import PromptTemplate
from IPython.display import Markdown, display
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate as PromptTemplateLangchain
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine

from typing import Optional, Union, Literal, List
from concurrent.futures import ThreadPoolExecutor
from functools import partial

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


class AsyncMultiQueryRAG:
    """
    Propuesta de Sistema RAG modularizado que permite cambiar fácilmente modelos de embeddings
    y rutas de almacenamiento vectorial.
    """

    def __init__(
        self,
        path_vector_storage: str,
        embedding_type: Literal["huggingface", "nomic"] = "huggingface",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 13,
        cross_encoder_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n=5,
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ):
        """
        Inicializa el sistema RAG.

        Args:
            path_vector_storage: Ruta donde están almacenados los vectores
            embedding_type: Tipo de embedding ("huggingface" o "nomic")
            embedding_model_name: Nombre del modelo de embedding
            top_k: Número de documentos similares a recuperar
            llm_model: Modelo de LLM a utilizar
            temperature: Temperatura para el LLM

        Modo de uso:
            object.query: De este modo puedes hacer consultas
        """
        self.path_vector_storage = path_vector_storage
        self.embedding_type = embedding_type
        self.embedding_model_name = embedding_model_name
        self.top_k = top_k
        self.cross_encoder_model_name = cross_encoder_model_name
        self.top_n = top_n
        self.llm_model = llm_model
        self.temperature = temperature

        # Variables de entorno
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.nomic_api_key = os.getenv("NOMIC_API_KEY")

        # Inicializar componentes | Inician con valor None pero luego cambian a la instancia.
        self.embed_model = None
        self.vector_index = None
        self.query_engine = None
        self.cross_encoder = None

        # Configurar el sistema
        self._setup_embeddings()
        self._setup_cross_encoder()

        self._setup_llm()
        self._load_vector_storage()
        # self._setup_query_engine()

    def _setup_embeddings(self):
        """Configura el modelo de embeddings según el tipo especificado."""
        if self.embedding_type == "huggingface":
            self.embed_model = HuggingFaceEmbedding(
                model_name=self.embedding_model_name
            )
        elif self.embedding_type == "nomic":
            if not self.nomic_api_key:
                raise ValueError(
                    "NOMIC_API_KEY no encontrada en las variables de entorno"
                )

            self.embed_model = NomicEmbedding(
                api_key=self.nomic_api_key,
                dimensionality=768,
                model_name=self.embedding_model_name or "nomic-embed-text-v1.5",
            )
        else:
            raise ValueError("embedding_type debe ser 'huggingface' o 'nomic'")

        # Configurar en Settings
        Settings.embed_model = self.embed_model

    def _setup_llm(self):
        """Configura el modelo de lenguaje."""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY no encontrada en las variables de entorno")

        Settings.llm = OpenAI(
            model=self.llm_model,
            temperature=self.temperature,
            api_key=self.openai_api_key,
        )

    def _load_vector_storage(self):
        """Carga el almacenamiento vectorial desde la ruta especificada."""
        try:
            vector_store = FaissVectorStore.from_persist_dir(self.path_vector_storage)

            storage_context = StorageContext.from_defaults(
                vector_store=vector_store, persist_dir=self.path_vector_storage
            )

            self.vector_index = load_index_from_storage(storage_context=storage_context)

        except Exception as e:
            raise Exception(f"Error al cargar el almacenamiento vectorial: {str(e)}")

    def _setup_cross_encoder(self):
        """Carga el modelo de re-rank"""
        print("Cargando Cross-Encoder...")
        self.cross_encoder = SentenceTransformerRerank(
            model=self.cross_encoder_model_name,
            top_n=self.top_n,
            keep_retrieval_score=True,
        )

    def multi_retrieval_embeddings(self, list_querys: List[str]):
        """En esta etapa, se realiza la inferencia al modelo de embeddings de manera simultánea"""
        # Ejecutar las llamadas en paralelo
        with ThreadPoolExecutor() as executor:
            # resultados = list(executor.map(retrieve_question, lista_preguntas))
            resultados_retrieval_embedding = list(
                # executor.map(self.get_retriever.retrieve, list_querys)self.vector_index
                executor.map(
                    self.vector_index.as_retriever(
                        similarity_top_k=self.top_k
                    ).retrieve,
                    list_querys,
                )
            )
        return resultados_retrieval_embedding

    def get_retriever(self):
        """Retorna el retriever para uso avanzado."""
        if not self.vector_index:
            raise Exception("El índice vectorial no está cargado")

        return self.vector_index.as_retriever(similarity_top_k=self.top_k)

    def update_top_k(self, new_top_k: int):
        """Actualiza el número de documentos a recuperar."""
        self.top_k = new_top_k
        self._setup_query_engine()

    def get_config(self) -> dict:
        """Retorna la configuración actual del sistema."""
        return {
            "path_vector_storage": self.path_vector_storage,
            "embedding_type": self.embedding_type,
            "embedding_model_name": self.embedding_model_name,
            "top_k": self.top_k,
            "llm_model": self.llm_model,
            "temperature": self.temperature,
        }

    def rerank_retrieval(self, querys, lista_nodes):
        return self.cross_encoder.postprocess_nodes(lista_nodes, query_str=querys)

    def query(self, query: str) -> str:
        """Este será el que llame a todo el sistema, como test"""
        lista_querys_generadas = multi_query_generator.generation(query)

        resultado_multi_retrieval = self.multi_retrieval_embeddings(
            lista_querys_generadas
        )

        lista_desempaquetada_nodes = []

        for i in tqdm(resultado_multi_retrieval, desc="Desempaquetando Lista"):
            lista_desempaquetada_nodes.extend(i)

        rerank_func = partial(
            self.rerank_retrieval, lista_nodes=lista_desempaquetada_nodes
        )

        # Ejecutar las llamadas en paralelo
        with ThreadPoolExecutor() as executor:
            # resultados = list(executor.map(retrieve_question, lista_preguntas))
            resultados_rerank_1 = list(
                executor.map(rerank_func, lista_querys_generadas)
            )

        # -------------Filtrar nodos Duplicados--------------------
        # Lista para almacenar nodos únicos
        filtered_nodes = []

        # Conjunto para rastrear combinaciones vistas de (file_name, page_label)
        seen = set()

        # Iterar sobre cada sublista en resultados_rerank
        for sublist in resultados_rerank_1:
            # Iterar sobre cada nodo en la sublista
            for node in sublist:
                # Extraer metadatos relevantes
                file_name = node.metadata.get("file_name", "")
                page_label = node.metadata.get("page_label", "")

                # Crear una tupla única para verificar duplicados
                key = (file_name, page_label)

                # Si la combinación no ha sido vista, agregar el nodo y marcar como visto
                if key not in seen:
                    seen.add(key)
                    filtered_nodes.append(node)
        print("Total de nodos filtrados:", len(filtered_nodes))

        retrieval_fake = CustomFakeRetrieval(filtered_nodes)

        query_engine = RetrieverQueryEngine.from_args(
            retrieval_fake, text_qa_template=PromptTemplate(RAG_QA_TEMPLATE)
        )

        response = query_engine.query(lista_querys_generadas[0])

        return response
