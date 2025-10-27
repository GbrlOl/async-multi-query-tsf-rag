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

from typing import Optional, Union, Literal

from prompts import RAG_QA_TEMPLATE

from dotenv import load_dotenv
import os

# Cargar el archivo .env
load_dotenv()

# Acceder a las variables de entorno
api_key = os.getenv("OPENAI_API_KEY")
nomic_api_key = os.getenv("NOMIC_API_KEY")


class NaiveRAG:
    """
    Sistema RAG modularizado que permite cambiar fácilmente modelos de embeddings
    y rutas de almacenamiento vectorial.
    """

    def __init__(
        self,
        path_vector_storage: str,
        embedding_type: Literal["huggingface", "nomic"] = "huggingface",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 13,
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
        """
        self.path_vector_storage = path_vector_storage
        self.embedding_type = embedding_type
        self.embedding_model_name = embedding_model_name
        self.top_k = top_k
        self.llm_model = llm_model
        self.temperature = temperature

        # Variables de entorno
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.nomic_api_key = os.getenv("NOMIC_API_KEY")

        # Inicializar componentes
        self.embed_model = None
        self.vector_index = None
        self.query_engine = None

        # Configurar el sistema
        self._setup_embeddings()
        self._setup_llm()
        self._load_vector_storage()
        self._setup_query_engine()

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

    def _setup_query_engine(self):
        """Configura el motor de consultas con el template personalizado."""

        qa_template = PromptTemplate(RAG_QA_TEMPLATE)

        self.query_engine = self.vector_index.as_query_engine(
            text_qa_template=qa_template, similarity_top_k=self.top_k
        )

    def query(self, query_text: str, display_response: bool = True) -> str:
        """
        Realiza una consulta al sistema RAG.

        Args:
            query_text: Texto de la consulta
            display_response: Si mostrar la respuesta formateada en Markdown

        Returns:
            Respuesta del sistema RAG
        """
        if not self.query_engine:
            raise Exception("El motor de consultas no está configurado")

        response = self.query_engine.query(query_text)

        return response

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
