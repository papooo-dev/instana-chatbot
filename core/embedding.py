"""
WatsonxEmbeddings 설정 및 초기화 모듈
IBM Watsonx.ai 임베딩 모델을 사용하여 텍스트를 벡터로 변환합니다.
"""
import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv

from langchain_ibm import WatsonxEmbeddings
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames

# 환경 변수 로드
load_dotenv()


class WatsonxEmbeddingManager:
    """WatsonxEmbeddings 관리 클래스"""
    
    def __init__(self, 
                 model_id: Optional[str] = None,
                 project_id: Optional[str] = None,
                 url: Optional[str] = None,
                 params: Optional[Dict[str, Any]] = None):
        """
        WatsonxEmbeddingManager 초기화
        
        Args:
            model_id: 사용할 임베딩 모델 ID
            project_id: Watson Studio 프로젝트 ID
            url: Watson Machine Learning 인스턴스 URL
            params: 모델 파라미터
        """
        self.model_id = model_id or os.getenv("WATSONX_MODEL_ID", "ibm/granite-embedding-107m-multilingual")
        self.project_id = project_id or os.getenv("WATSONX_PROJECT_ID")
        self.url = url or os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
        
        # 기본 파라미터 설정
        self.params = params or {
            EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 512,
            EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
        }
        
        self.embeddings = None
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """WatsonxEmbeddings 인스턴스 초기화"""
        try:
            if not self.project_id:
                raise ValueError("WATSONX_PROJECT_ID 환경 변수가 설정되지 않았습니다.")
            
            self.embeddings = WatsonxEmbeddings(
                model_id=self.model_id,
                url=self.url,
                project_id=self.project_id,
                params=self.params,
            )
            
            print(f"WatsonxEmbeddings 초기화 완료:")
            print(f"  - 모델: {self.model_id}")
            print(f"  - URL: {self.url}")
            print(f"  - 프로젝트 ID: {self.project_id}")
            
        except Exception as e:
            raise Exception(f"WatsonxEmbeddings 초기화 실패: {e}")
    
    def get_embeddings(self) -> WatsonxEmbeddings:
        """임베딩 인스턴스 반환"""
        if self.embeddings is None:
            raise RuntimeError("WatsonxEmbeddings가 초기화되지 않았습니다.")
        return self.embeddings
    
    def test_embedding(self, text: str = "테스트 텍스트입니다.") -> list:
        """
        임베딩 기능 테스트
        
        Args:
            text: 테스트할 텍스트
            
        Returns:
            임베딩 벡터
        """
        try:
            embedding = self.embeddings.embed_query(text)
            print(f"임베딩 테스트 성공: {len(embedding)}차원 벡터 생성")
            print(f"벡터 샘플 (처음 5개): {embedding[:5]}")
            return embedding
            
        except Exception as e:
            raise Exception(f"임베딩 테스트 실패: {e}")
    
    def embed_documents(self, texts: list) -> list:
        """
        여러 문서의 임베딩 생성
        
        Args:
            texts: 임베딩할 텍스트 리스트
            
        Returns:
            임베딩 벡터 리스트
        """
        try:
            embeddings = self.embeddings.embed_documents(texts)
            print(f"{len(texts)}개 문서 임베딩 완료")
            return embeddings
            
        except Exception as e:
            raise Exception(f"문서 임베딩 실패: {e}")
    
    def embed_query(self, text: str) -> list:
        """
        단일 쿼리 텍스트의 임베딩 생성
        
        Args:
            text: 임베딩할 텍스트
            
        Returns:
            임베딩 벡터
        """
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
            
        except Exception as e:
            raise Exception(f"쿼리 임베딩 실패: {e}")


def create_watsonx_embeddings() -> WatsonxEmbeddingManager:
    """
    WatsonxEmbeddingManager 인스턴스 생성
    
    Returns:
        WatsonxEmbeddingManager 인스턴스
    """
    return WatsonxEmbeddingManager()


# 환경 변수 검증 함수
def validate_watsonx_config() -> bool:
    """
    Watsonx 설정이 올바른지 검증
    
    Returns:
        설정이 유효한지 여부
    """
    required_vars = ["WATSONX_PROJECT_ID"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"누락된 환경 변수: {missing_vars}")
        print("다음 환경 변수들을 설정해주세요:")
        print("  - WATSONX_PROJECT_ID: Watson Studio 프로젝트 ID")
        print("  - WATSONX_URL: Watson Machine Learning 인스턴스 URL (선택사항)")
        print("  - WATSONX_MODEL_ID: 사용할 임베딩 모델 ID (선택사항)")
        return False
    
    return True
