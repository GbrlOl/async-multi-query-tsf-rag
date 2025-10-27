from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from prompts import LLM_EVALUATOR_TEMPLATE


class LLMEvaluator:
    def __init__(self, model_llm_id: str = "gpt-4o-mini", temperature: float = 0.0):
        """
        Esta función recibe tres parámetros de entrada:

        1. La consulta sobre el parámetro (query).
        2. La respuesta que genera el sistema RAG (respuesta_rag).
        3. La respuesta de referencia o ground truth (respuesta_referencia)

        Nota: Estoy utilizando GPT-4o-mini, dado que GPT-3.5 no evalúa bien.

        Este sistema retorna 1 si la respuesta del sistema RAG es correcta con respecto al ground truth, 0 caso contrario.
        """
        self.model_llm_id = model_llm_id
        self.temperature = temperature

    def evaluation(
        self, query: str, respuesta_rag: str, respuesta_referencia: str
    ) -> str:

        prompt_template_evaluador = PromptTemplate.from_template(LLM_EVALUATOR_TEMPLATE)

        llm = ChatOpenAI(model=self.model_llm_id, temperature=self.temperature)

        chain = prompt_template_evaluador | llm | StrOutputParser()

        response = chain.invoke(
            {
                "pregunta": query,
                "respuesta_rag": respuesta_rag,
                "ground_truth": respuesta_referencia,
            }
        )

        return response
