"""
LLM wiring for IBM watsonx via LangChain.
"""
import os
from typing import Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_ibm import WatsonxLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

from langchain_openai import ChatOpenAI
from prompts import SYSTEM_PROMPT

def build_llm() -> BaseLanguageModel:
    """
    Create a LangChain LLM backed by IBM watsonx.
    Requires env: WATSONX_APIKEY, WATSONX_URL, WATSONX_PROJECT_ID, WATSONX_MODEL_ID
    """
    api_key = os.getenv("WATSONX_APIKEY")
    url = os.getenv("WATSONX_URL")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    model_id = os.getenv("WATSONX_MODEL_ID", "ibm/granite-20b-multilingual")

    if not all([api_key, url, project_id]):
        raise RuntimeError("Missing watsonx env vars (WATSONX_APIKEY, WATSONX_URL, WATSONX_PROJECT_ID).")

    parameters = {
        # GenTextParamsMetaNames.DECODING_METHOD: "sample",
        GenTextParamsMetaNames.MAX_NEW_TOKENS: 800,
        GenTextParamsMetaNames.MIN_NEW_TOKENS: 1,
        GenTextParamsMetaNames.TEMPERATURE: 0.5,
        # GenTextParamsMetaNames.TOP_K: 50,
        # GenTextParamsMetaNames.TOP_P: 1,
    }

    # See langchain-ibm docs for all config kwargs
    llm = WatsonxLLM(
        model_id=model_id,
        url=url,
        project_id=project_id,
        # common generation params (tune as needed)
        # params={
        #     "decoding_method": "greedy",
        #     "max_new_tokens": 512,
        #     "temperature": 0.2,
        #     "top_p": 1.0,
        # },
        params = parameters 
    )

    return llm


def build_chain():
    """
    Build a simple chat chain with a system prompt and a placeholder for chat history.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )
    llm = build_llm()
    print("::: llm :::", llm)
    parser = StrOutputParser()
    return prompt | llm | parser


def build_streaming_chain():
    """
    Build a streaming chat chain with a system prompt and a placeholder for chat history.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )
    llm = build_llm()
    print("::: llm :::", llm)
    parser = StrOutputParser()
    return prompt | llm | parser
