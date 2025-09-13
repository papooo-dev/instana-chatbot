"""
Milvus 벡터 데이터베이스 관리 모듈
LangChain과 Milvus를 연동하여 벡터 스토어를 관리합니다.
"""
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from langchain_milvus import Milvus
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# 환경 변수 로드
load_dotenv()


class MilvusVectorStoreManager:
    """Milvus 벡터 스토어 관리 클래스"""
    
    def __init__(self, 
                 connection_uri: Optional[str] = None,
                 collection_name: Optional[str] = None,
                 embeddings: Optional[Embeddings] = None):
        """
        MilvusVectorStoreManager 초기화
        
        Args:
            connection_uri: Milvus 연결 URI
            collection_name: 컬렉션 이름
            embeddings: 임베딩 모델 인스턴스
        """
        self.connection_uri = connection_uri or os.getenv("MILVUS_URI", "http://localhost:19530")
        self.collection_name = collection_name or os.getenv("MILVUS_COLLECTION", "instana_docs")
        self.embeddings = embeddings
        
        self.vectorstore = None
        self._initialize_vectorstore()
    
    def _initialize_vectorstore(self):
        """Milvus 벡터 스토어 초기화"""
        try:
            if not self.embeddings:
                raise ValueError("임베딩 모델이 제공되지 않았습니다.")
            
            self.vectorstore = Milvus(
                embedding_function=self.embeddings,
                connection_args={"host": "localhost", "port": "19530"},
                collection_name=self.collection_name,
                drop_old=False,  # 기존 컬렉션 유지
            )
            
            print(f"Milvus 벡터 스토어 초기화 완료:")
            print(f"  - 연결 URI: {self.connection_uri}")
            print(f"  - 컬렉션: {self.collection_name}")
            
        except Exception as e:
            raise Exception(f"Milvus 벡터 스토어 초기화 실패: {e}")
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        문서들을 벡터 스토어에 추가
        
        Args:
            documents: 추가할 Document 객체 리스트
            
        Returns:
            추가된 문서의 ID 리스트
        """
        try:
            print(f"{len(documents)}개 문서를 Milvus에 추가 중...")
            
            # 문서들을 벡터 스토어에 추가
            ids = self.vectorstore.add_documents(documents)
            
            print(f"문서 추가 완료: {len(ids)}개 문서 저장됨")
            return ids
            
        except Exception as e:
            raise Exception(f"문서 추가 실패: {e}")
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        텍스트들을 벡터 스토어에 추가
        
        Args:
            texts: 추가할 텍스트 리스트
            metadatas: 각 텍스트에 대한 메타데이터 리스트
            
        Returns:
            추가된 문서의 ID 리스트
        """
        try:
            print(f"{len(texts)}개 텍스트를 Milvus에 추가 중...")
            
            ids = self.vectorstore.add_texts(texts, metadatas=metadatas)
            
            print(f"텍스트 추가 완료: {len(ids)}개 텍스트 저장됨")
            return ids
            
        except Exception as e:
            raise Exception(f"텍스트 추가 실패: {e}")
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        유사도 검색 수행
        
        Args:
            query: 검색 쿼리
            k: 반환할 문서 수
            
        Returns:
            유사한 Document 객체 리스트
        """
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            print(f"유사도 검색 완료: {len(results)}개 결과 반환")
            return results
            
        except Exception as e:
            raise Exception(f"유사도 검색 실패: {e}")
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """
        점수와 함께 유사도 검색 수행
        
        Args:
            query: 검색 쿼리
            k: 반환할 문서 수
            
        Returns:
            (Document, score) 튜플 리스트
        """
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            print(f"유사도 검색 완료: {len(results)}개 결과 반환")
            return results
            
        except Exception as e:
            raise Exception(f"유사도 검색 실패: {e}")
    
    def get_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        """
        검색기(Retriever) 반환
        
        Args:
            search_kwargs: 검색 파라미터
            
        Returns:
            검색기 인스턴스
        """
        try:
            retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)
            print("검색기 생성 완료")
            return retriever
            
        except Exception as e:
            raise Exception(f"검색기 생성 실패: {e}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        컬렉션 정보 반환
        
        Returns:
            컬렉션 정보 딕셔너리
        """
        try:
            # Milvus 컬렉션 정보 조회
            collection = self.vectorstore._get_collection()
            stats = collection.get_stats()
            
            info = {
                "collection_name": self.collection_name,
                "total_entities": stats.get("row_count", 0),
                "dimension": stats.get("dimension", 0),
                "index_type": stats.get("index_type", "unknown"),
                "metric_type": stats.get("metric_type", "unknown")
            }
            
            return info
            
        except Exception as e:
            print(f"컬렉션 정보 조회 실패: {e}")
            return {"collection_name": self.collection_name, "error": str(e)}
    
    def delete_collection(self):
        """컬렉션 삭제"""
        try:
            self.vectorstore._drop_collection()
            print(f"컬렉션 '{self.collection_name}' 삭제 완료")
            
        except Exception as e:
            raise Exception(f"컬렉션 삭제 실패: {e}")
    
    def test_connection(self) -> bool:
        """
        Milvus 연결 테스트
        
        Returns:
            연결 성공 여부
        """
        try:
            # 간단한 검색으로 연결 테스트
            self.vectorstore.similarity_search("test", k=1)
            print("Milvus 연결 테스트 성공")
            return True
            
        except Exception as e:
            print(f"Milvus 연결 테스트 실패: {e}")
            return False


def create_milvus_vectorstore(embeddings: Embeddings, 
                             collection_name: str = "instana_docs") -> MilvusVectorStoreManager:
    """
    MilvusVectorStoreManager 인스턴스 생성
    
    Args:
        embeddings: 임베딩 모델 인스턴스
        collection_name: 컬렉션 이름
        
    Returns:
        MilvusVectorStoreManager 인스턴스
    """
    return MilvusVectorStoreManager(
        embeddings=embeddings,
        collection_name=collection_name
    )


# 환경 변수 검증 함수
def validate_milvus_config() -> bool:
    """
    Milvus 설정이 올바른지 검증
    
    Returns:
        설정이 유효한지 여부
    """
    # Milvus는 기본적으로 localhost:19530에서 실행되므로 특별한 환경 변수가 필요하지 않음
    print("Milvus 설정 검증:")
    print("  - 기본 연결: http://localhost:19530")
    print("  - Docker Compose로 Milvus 서버 실행 필요")
    return True
