"""
Milvus 벡터 데이터베이스 관리 모듈 (MilvusClient 직접 사용)
Instana 메트릭 수집을 위해 pymilvus.MilvusClient를 직접 사용합니다.
"""
import os
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

from pymilvus import MilvusClient, DataType
from traceloop.sdk.decorators import task, workflow
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# 환경 변수 로드
load_dotenv()


class MilvusVectorStoreManager:
    """Milvus 벡터 스토어 관리 클래스 (MilvusClient 직접 사용)"""
    
    def __init__(self, 
                 connection_uri: Optional[str] = None,
                 collection_name: Optional[str] = None,
                 embeddings: Optional[Embeddings] = None,
                 dimension: int = 1024):
        """
        MilvusVectorStoreManager 초기화
        
        Args:
            connection_uri: Milvus 연결 URI
            collection_name: 컬렉션 이름
            embeddings: 임베딩 모델 인스턴스
            dimension: 벡터 차원 (multilingual-e5-large는 1024)
        """
        self.connection_uri = connection_uri or os.getenv("MILVUS_URI", "http://localhost:19530")
        self.collection_name = collection_name or os.getenv("MILVUS_COLLECTION", "instana_docs")
        self.embeddings = embeddings
        self.dimension = dimension
        
        # MilvusClient 초기화
        self.client: Optional[MilvusClient] = None
        self._initialize_client()
    
    @task(name="milvus_setup_client")  # pyright: ignore[reportArgumentType]
    def _initialize_client(self):
        """Milvus 클라이언트 초기화"""
        try:
            if not self.embeddings:
                raise ValueError("임베딩 모델이 제공되지 않았습니다.")
            
            # MilvusClient 생성
            self.client = MilvusClient(uri=self.connection_uri)
            
            # 컬렉션이 없으면 생성
            if not self.client.has_collection(collection_name=self.collection_name):
                self._create_collection()
            
            print(f"Milvus 클라이언트 초기화 완료:")
            print(f"  - 연결 URI: {self.connection_uri}")
            print(f"  - 컬렉션: {self.collection_name}")
            print(f"  - 벡터 차원: {self.dimension}")
            
        except Exception as e:
            raise Exception(f"Milvus 클라이언트 초기화 실패: {e}")
    
    def _create_collection(self):
        """컬렉션 생성"""
        try:
            # 스키마 정의
            schema = self.client.create_schema(
                auto_id=True,
                enable_dynamic_field=True
            )
            
            # 필드 추가
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
            schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.dimension)
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="page", datatype=DataType.INT64)
            schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=512)
            
            # 인덱스 파라미터
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_type="IVF_FLAT",
                metric_type="COSINE",
                params={"nlist": 128}
            )
            
            # 컬렉션 생성
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params
            )
            
            print(f"컬렉션 '{self.collection_name}' 생성 완료")
            
        except Exception as e:
            raise Exception(f"컬렉션 생성 실패: {e}")
    
    @task(name="milvus_insert_documents")  # pyright: ignore[reportArgumentType]
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
            
            # 문서 텍스트 추출
            texts = [doc.page_content for doc in documents]
            
            # 임베딩 생성
            vectors = self.embeddings.embed_documents(texts)
            
            # 데이터 준비
            data = []
            for i, (doc, vector) in enumerate(zip(documents, vectors)):
                data.append({
                    "vector": vector,
                    "text": doc.page_content,
                    "page": doc.metadata.get("page", 0),
                    "source": doc.metadata.get("source", "unknown")
                })
            
            # Milvus에 삽입
            result = self.client.insert(
                collection_name=self.collection_name,
                data=data
            )
            
            print(f"문서 추가 완료: {len(result['ids'])}개 문서 저장됨")
            return [str(id) for id in result['ids']]
            
        except Exception as e:
            raise Exception(f"문서 추가 실패: {e}")
    
    @task(name="milvus_insert_texts")  # pyright: ignore[reportArgumentType]
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
            
            # 임베딩 생성
            vectors = self.embeddings.embed_documents(texts)
            
            # 데이터 준비
            data = []
            for i, (text, vector) in enumerate(zip(texts, vectors)):
                metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                data.append({
                    "vector": vector,
                    "text": text,
                    "page": metadata.get("page", 0),
                    "source": metadata.get("source", "unknown")
                })
            
            # Milvus에 삽입
            result = self.client.insert(
                collection_name=self.collection_name,
                data=data
            )
            
            print(f"텍스트 추가 완료: {len(result['ids'])}개 텍스트 저장됨")
            return [str(id) for id in result['ids']]
            
        except Exception as e:
            raise Exception(f"텍스트 추가 실패: {e}")
    
    @task(name="milvus_vector_search")  # pyright: ignore[reportArgumentType]
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
            # 쿼리 임베딩 생성
            query_vector = self.embeddings.embed_query(query)
            
            # 검색 수행
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_vector],
                limit=k,
                output_fields=["text", "page", "source"]
            )
            
            # Document 객체로 변환
            documents = []
            for hits in results:
                for hit in hits:
                    doc = Document(
                        page_content=hit['entity']['text'],
                        metadata={
                            'page': hit['entity']['page'],
                            'source': hit['entity']['source'],
                            'score': hit['distance']
                        }
                    )
                    documents.append(doc)
            
            print(f"유사도 검색 완료: {len(documents)}개 결과 반환")
            return documents
            
        except Exception as e:
            raise Exception(f"유사도 검색 실패: {e}")
    
    @task(name="milvus_vector_search_with_score")  # pyright: ignore[reportArgumentType]
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        점수와 함께 유사도 검색 수행
        
        Args:
            query: 검색 쿼리
            k: 반환할 문서 수
            
        Returns:
            (Document, score) 튜플 리스트
        """
        try:
            # 쿼리 임베딩 생성
            query_vector = self.embeddings.embed_query(query)
            
            # 검색 수행
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_vector],
                limit=k,
                output_fields=["text", "page", "source"]
            )
            
            # (Document, score) 튜플로 변환
            documents_with_scores = []
            for hits in results:
                for hit in hits:
                    doc = Document(
                        page_content=hit['entity']['text'],
                        metadata={
                            'page': hit['entity']['page'],
                            'source': hit['entity']['source']
                        }
                    )
                    score = hit['distance']
                    documents_with_scores.append((doc, score))
            
            print(f"유사도 검색 완료: {len(documents_with_scores)}개 결과 반환")
            return documents_with_scores
            
        except Exception as e:
            raise Exception(f"유사도 검색 실패: {e}")
    
    @task(name="milvus_query_collection_info")  # pyright: ignore[reportArgumentType]
    def get_collection_info(self) -> Dict[str, Any]:
        """
        컬렉션 정보 반환
        
        Returns:
            컬렉션 정보 딕셔너리
        """
        try:
            # 컬렉션 통계 조회
            stats = self.client.get_collection_stats(collection_name=self.collection_name)
            
            info = {
                "collection_name": self.collection_name,
                "total_entities": stats.get("row_count", 0),
                "dimension": self.dimension
            }
            
            return info
            
        except Exception as e:
            print(f"컬렉션 정보 조회 실패: {e}")
            return {"collection_name": self.collection_name, "error": str(e)}
    
    @task(name="milvus_delete_entities")  # pyright: ignore[reportArgumentType]
    def delete_by_ids(self, ids: List[str]):
        """ID로 엔티티 삭제"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                ids=[int(id) for id in ids]
            )
            print(f"{len(ids)}개 엔티티 삭제 완료")
            
        except Exception as e:
            raise Exception(f"엔티티 삭제 실패: {e}")
    
    @task(name="milvus_delete_collection")  # pyright: ignore[reportArgumentType]
    def delete_collection(self):
        """컬렉션 삭제"""
        try:
            self.client.drop_collection(collection_name=self.collection_name)
            print(f"컬렉션 '{self.collection_name}' 삭제 완료")
            
        except Exception as e:
            raise Exception(f"컬렉션 삭제 실패: {e}")
    
    @task(name="milvus_test_connection")  # pyright: ignore[reportArgumentType]
    def test_connection(self) -> bool:
        """
        Milvus 연결 테스트
        
        Returns:
            연결 성공 여부
        """
        try:
            # 컬렉션 존재 여부 확인
            has_collection = self.client.has_collection(collection_name=self.collection_name)
            print(f"Milvus 연결 테스트 성공 (컬렉션 존재: {has_collection})")
            return True
            
        except Exception as e:
            print(f"Milvus 연결 테스트 실패: {e}")
            return False


def create_milvus_vectorstore(embeddings: Embeddings, 
                             collection_name: str = "instana_docs",
                             dimension: int = 1024) -> MilvusVectorStoreManager:
    """
    MilvusVectorStoreManager 인스턴스 생성
    
    Args:
        embeddings: 임베딩 모델 인스턴스
        collection_name: 컬렉션 이름
        dimension: 벡터 차원
        
    Returns:
        MilvusVectorStoreManager 인스턴스
    """
    return MilvusVectorStoreManager(
        embeddings=embeddings,
        collection_name=collection_name,
        dimension=dimension
    )


# 환경 변수 검증 함수
def validate_milvus_config() -> bool:
    """
    Milvus 설정이 올바른지 검증
    
    Returns:
        설정이 유효한지 여부
    """
    print("Milvus 설정 검증:")
    print("  - 기본 연결: http://localhost:19530")
    print("  - Docker Compose로 Milvus 서버 실행 필요")
    return True

# Made with Bob
