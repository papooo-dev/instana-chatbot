"""
LLM wiring for IBM watsonx via LangChain with RAG integration.
"""
import os
from typing import Optional, Dict, Any

from langchain_core.language_models import BaseLanguageModel
from langchain_ibm import ChatWatsonx
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from .prompts import SYSTEM_PROMPT_WITH_RAG
from .rag import create_rag_system

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
        "temperature": 0.1,
        "max_tokens": 800,
    }

    # See langchain-ibm docs for all config kwargs
    llm = ChatWatsonx(
        model_id=model_id,
        url=url,
        project_id=project_id,
        params = parameters 
    )

    return llm


def build_streaming_chain():
    """
    Build a streaming chat chain with RAG integration.
    Retrieves relevant documents and includes them in the context.
    """
    def enhance_query_with_context(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """사용자 입력을 RAG 컨텍스트로 강화"""
        user_input = inputs["input"]
        
        # RAG 컨텍스트 생성
        context_info = get_rag_context(user_input)
        
        # 강화된 입력 생성
        enhanced_input = f"""
사용자 질문: {user_input}

관련 문서 정보:
{context_info['context']}

위의 관련 문서 정보를 바탕으로 사용자 질문에 정확하고 도움이 되는 답변을 제공해주세요.
답변할 때는 관련 문서의 내용을 참조하여 구체적이고 정확한 정보를 제공하되, 
문서에 없는 내용은 추측하지 말고 "문서에 해당 정보가 없습니다"라고 명시해주세요.
"""
        
        return {
            "input": enhanced_input,
            "history": inputs["history"],
            "rag_context": context_info
        }
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT_WITH_RAG),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )
    
    llm = build_llm()
    parser = StrOutputParser()
    
    # 체인 구성: 입력 강화 -> 프롬프트 -> LLM -> 파서
    return enhance_query_with_context | prompt | llm | parser


def get_rag_context(query: str) -> Dict[str, Any]:
    """
    Get RAG context for a given query.
    
    Args:
        query: User query
        
    Returns:
        Dictionary containing context and source information
    """
    try:
        # RAG 시스템 초기화 (싱글톤 패턴으로 최적화 가능)
        rag_system = create_rag_system()
        
        # 상세 컨텍스트 생성
        context_info = rag_system.get_detailed_context(query)
        
        return context_info
        
    except Exception as e:
        print(f"RAG 컨텍스트 생성 실패: {e}")
        return {
            "context": "관련 문서를 찾을 수 없습니다.",
            "sources": [],
            "total_documents": 0,
            "average_score": 0.0
        }


