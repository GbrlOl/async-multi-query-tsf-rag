from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from prompts import ASYNC_MULTI_QUERY_TEMPLATE_V1, ASYNC_MULTI_QUERY_TEMPLATE_V2
from typing import List


class MultiQueryGeneration:
    def __init__(
        self,
        model_llm_id: str = "gpt-4o-mini",
        temperature: float = 0.0,
        num_query: int = 4,
    ):
        """
        Inicializa el Generador de Múltiples Consultas.

        Args:
            model_llm_id: ID del LLM de OpenAI
            temperature: Temperatura
            embedding_model_name: Nombre del modelo de embedding
            num_query: Número de consultas para que el LLM genere.
        Nota:
            Por defecto generará 4 consultas y se agrega la principal, resultando en 5 consultas en total.
        Advertencia:
            El LLM puede fallar en generar la cantidad de consultas solicitadas, recuerde que son malos razonando matemáticamente
        """
        self.model_llm_id = model_llm_id
        self.temperature = temperature
        self.num_query = num_query

    def generation(self, query: str) -> List[str]:

        messages = [
            ("system", ASYNC_MULTI_QUERY_TEMPLATE_V2.format(num_query=self.num_query)),
            ("user", query),
        ]

        llm_generador = ChatOpenAI(name="gpt-4o-mini", temperature=0.0)

        llm_response = llm_generador.invoke(messages)

        lista_preguntas = llm_response.content.split(
            "||"
        )  # Acá separo las querys y las guardo en una lista

        lista_preguntas.insert(0, query)

        return lista_preguntas
