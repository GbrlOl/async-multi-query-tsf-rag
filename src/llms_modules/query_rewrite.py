# from langchain_core.prompts import
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from prompts import QUERY_REWRITE_TEMPLATE


class QueryRewrite:
    def __init__(
        self, model_llm_id: str = "gpt-3.5-turbo-1106", temperature: float = 0.0
    ):
        self.model_llm_id = model_llm_id
        self.temperature = temperature

    def query(self, query: str) -> str:

        messages = [("system", QUERY_REWRITE_TEMPLATE), ("user", query)]

        llm = (
            ChatOpenAI(model=self.model_llm_id, temperature=self.temperature)
            | StrOutputParser()
        )

        response_llm = llm.invoke(messages)
        return response_llm
