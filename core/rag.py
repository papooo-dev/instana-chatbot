"""
RAG (Retrieval-Augmented Generation) 시스템
Milvus 벡터 스토어와 Watsonx LLM을 통합한 완전한 RAG 시스템
"""
import os
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from .embedding import WatsonxEmbeddingManager
from .milvus_manager import MilvusVectorStoreManager

# 환경 변수 로드
load_dotenv()


class InstanaRAGSystem:
    """Instana 문서 기반 RAG 시스템"""
    
    def __init__(self, 
                 collection_name: str = "instana_docs",
                 top_k: int = 10,
                 similarity_threshold: float = 0.4):
        """
        RAG 시스템 초기화
        
        Args:
            collection_name: Milvus 컬렉션 이름
            top_k: 검색할 문서 수
            similarity_threshold: 유사도 임계값
        """
        self.collection_name = collection_name
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        
        # 임베딩 매니저 초기화
        self.embedding_manager = WatsonxEmbeddingManager(
            model_id="ibm/granite-embedding-107m-multilingual"
        )
        
        # 벡터 스토어 매니저 초기화
        self.vectorstore_manager = MilvusVectorStoreManager(
            embeddings=self.embedding_manager.get_embeddings(),
            collection_name=collection_name
        )
        
        print(f"RAG 시스템 초기화 완료:")
        print(f"  - 컬렉션: {collection_name}")
        print(f"  - 검색 문서 수: {top_k}")
        print(f"  - 유사도 임계값: {similarity_threshold}")
    
    def search_documents(self, query: str) -> List[Document]:
        """
        쿼리에 대한 관련 문서 검색
        
        Args:
            query: 검색 쿼리
            
        Returns:
            관련 문서 리스트
        """
        try:
            # 유사도 검색 수행 (더 많은 결과를 위해 k 값을 증가)
            results = self.vectorstore_manager.similarity_search(query, k=10)
            
            print(f"'{query}' 검색 결과: {len(results)}개 문서 발견")
            return results
            
        except Exception as e:
            print(f"문서 검색 실패: {e}")
            return []
    
    def search_with_scores(self, query: str) -> List[Tuple[Document, float]]:
        """
        점수와 함께 문서 검색
        
        Args:
            query: 검색 쿼리
            
        Returns:
            (문서, 점수) 튜플 리스트
        """
        try:
            results = self.vectorstore_manager.similarity_search_with_score(query, k=self.top_k)
            
            print(f"'{query}' 원본 검색 결과: {len(results)}개 문서")
            
            # 유사도 임계값 필터링 (임계값을 낮춰서 더 많은 결과 포함)
            filtered_results = [
                (doc, score) for doc, score in results 
                if score >= 0.3  # 임계값을 0.3으로 낮춤
            ]
            
            print(f"'{query}' 필터링 후 결과: {len(filtered_results)}개 문서 (임계값: 0.3)")
            return filtered_results
            
        except Exception as e:
            print(f"문서 검색 실패: {e}")
            return []
    
    def get_context(self, query: str) -> str:
        """
        쿼리에 대한 컨텍스트 생성
        
        Args:
            query: 검색 쿼리
            
        Returns:
            컨텍스트 문자열
        """
        try:
            # 관련 문서 검색
            documents = self.search_documents(query)
            
            if not documents:
                return "관련 문서를 찾을 수 없습니다."
            
            # 컨텍스트 구성
            context_parts = []
            for i, doc in enumerate(documents, 1):
                # 문서 내용 요약 (너무 길면 잘라내기)
                content = doc.page_content
                if len(content) > 500:
                    content = content[:500] + "..."
                
                # 메타데이터에서 페이지 정보 추출
                page_info = ""
                if 'page' in doc.metadata:
                    page_info = f" (페이지 {doc.metadata['page']})"
                
                context_parts.append(f"[문서 {i}{page_info}]\n{content}")
            
            context = "\n\n".join(context_parts)
            
            print(f"컨텍스트 생성 완료: {len(context)}자")
            return context
            
        except Exception as e:
            print(f"컨텍스트 생성 실패: {e}")
            return "컨텍스트를 생성할 수 없습니다."
    
    def get_detailed_context(self, query: str) -> Dict[str, Any]:
        """
        상세한 컨텍스트 정보 반환
        
        Args:
            query: 검색 쿼리
            
        Returns:
            상세 컨텍스트 딕셔너리
        """
        try:
            # 점수와 함께 검색
            results = self.search_with_scores(query)
            
            if not results:
                return {
                    "context": "관련 문서를 찾을 수 없습니다.",
                    "sources": [],
                    "total_documents": 0,
                    "average_score": 0.0
                }
            
            # 컨텍스트 구성
            context_parts = []
            sources = []
            total_score = 0.0
            
            for i, (doc, score) in enumerate(results, 1):
                # 문서 내용
                content = doc.page_content
                if len(content) > 400:
                    content = content[:400] + "..."
                
                # 소스 정보
                source_info = {
                    "document_id": i,
                    "page": doc.metadata.get('page', 'N/A'),
                    "chunk_id": doc.metadata.get('chunk_id', 'N/A'),
                    "score": round(score, 4),
                    "content_preview": content[:100] + "..." if len(content) > 100 else content
                }
                sources.append(source_info)
                
                # 컨텍스트에 추가
                page_info = f" (페이지 {doc.metadata.get('page', 'N/A')}, 점수: {score:.3f})"
                context_parts.append(f"[문서 {i}{page_info}]\n{content}")
                
                total_score += score
            
            context = "\n\n".join(context_parts)
            average_score = total_score / len(results)
            
            return {
                "context": context,
                "sources": sources,
                "total_documents": len(results),
                "average_score": round(average_score, 4)
            }
            
        except Exception as e:
            print(f"상세 컨텍스트 생성 실패: {e}")
            return {
                "context": "컨텍스트를 생성할 수 없습니다.",
                "sources": [],
                "total_documents": 0,
                "average_score": 0.0
            }
    
    def test_rag_system(self) -> bool:
        """
        RAG 시스템 테스트
        
        Returns:
            테스트 성공 여부
        """
        try:
            print("🔍 RAG 시스템 테스트 시작...")
            
            # 테스트 쿼리들
            test_queries = [
                "Instana란 무엇인가요?",
                "애플리케이션 성능 모니터링",
                "IBM의 관찰 가능성 솔루션",
                "마이크로서비스 모니터링"
            ]
            
            for query in test_queries:
                print(f"\n📝 테스트 쿼리: '{query}'")
                
                # 문서 검색 테스트
                documents = self.search_documents(query)
                if not documents:
                    print(f"❌ '{query}' 검색 실패")
                    return False
                
                print(f"✅ {len(documents)}개 문서 검색 성공")
                
                # 컨텍스트 생성 테스트
                context = self.get_context(query)
                if len(context) < 50:
                    print(f"❌ '{query}' 컨텍스트 생성 실패")
                    return False
                
                print(f"✅ 컨텍스트 생성 성공 ({len(context)}자)")
            
            print("\n🎉 RAG 시스템 테스트 완료!")
            return True
            
        except Exception as e:
            print(f"❌ RAG 시스템 테스트 실패: {e}")
            return False


class InstanaRetriever(BaseRetriever):
    """LangChain 호환 검색기"""
    
    def __init__(self, rag_system: InstanaRAGSystem):
        super().__init__()
        self.rag_system = rag_system
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """관련 문서 검색"""
        return self.rag_system.search_documents(query)


def create_rag_system(collection_name: str = "instana_docs") -> InstanaRAGSystem:
    """
    RAG 시스템 인스턴스 생성
    
    Args:
        collection_name: Milvus 컬렉션 이름
        
    Returns:
        InstanaRAGSystem 인스턴스
    """
    return InstanaRAGSystem(collection_name=collection_name)


def create_retriever(rag_system: InstanaRAGSystem) -> InstanaRetriever:
    """
    LangChain 호환 검색기 생성
    
    Args:
        rag_system: RAG 시스템 인스턴스
        
    Returns:
        InstanaRetriever 인스턴스
    """
    return InstanaRetriever(rag_system)
