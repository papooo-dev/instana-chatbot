"""
Instana PDF 문서를 Milvus 벡터 데이터베이스에 저장하는 메인 스크립트
"""
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pdf_processor import PDFProcessor
from core.embedding import WatsonxEmbeddingManager, validate_watsonx_config
from core.milvus_manager import MilvusVectorStoreManager, validate_milvus_config


def main():
    print("=" * 60)
    print("Instana PDF 문서를 Milvus 벡터 DB에 저장하는 스크립트")
    print("=" * 60)
    
    # PDF 파일 경로 설정
    pdf_path = "data/instana-observability-1.0.301-documentation.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF 파일을 찾을 수 없습니다: {pdf_path}")
        print("현재 디렉토리에 PDF 파일이 있는지 확인해주세요.")
        return False
    
    try:
        # 1. 환경 설정 검증
        print("\n1️⃣ 환경 설정 검증 중...")
        
        if not validate_watsonx_config():
            print("❌ Watsonx 설정이 올바르지 않습니다.")
            return False
        
        if not validate_milvus_config():
            print("❌ Milvus 설정이 올바르지 않습니다.")
            return False
        
        print("✅ 환경 설정 검증 완료")
        
        # 2. PDF 처리
        print("\n2️⃣ PDF 문서 처리 중...")
        pdf_processor = PDFProcessor(chunk_size=1000, chunk_overlap=200)
        documents = pdf_processor.process_pdf(pdf_path)
        
        # 문서 통계 출력
        stats = pdf_processor.get_document_stats(documents)
        print(f"📊 문서 통계:")
        print(f"   - 총 청크 수: {stats['total_chunks']}")
        print(f"   - 총 문자 수: {stats['total_characters']:,}")
        print(f"   - 평균 청크 크기: {stats['avg_chunk_size']}")
        print(f"   - 최소 청크 크기: {stats['min_chunk_size']}")
        print(f"   - 최대 청크 크기: {stats['max_chunk_size']}")
        
        # 3. Watsonx 임베딩 초기화
        print("\n3️⃣ Watsonx 임베딩 모델 초기화 중...")
        # 명시적으로 임베딩 모델 설정
        embedding_manager = WatsonxEmbeddingManager(
            model_id="ibm/granite-embedding-107m-multilingual"
        )
        
        # 임베딩 테스트
        print("🔍 임베딩 기능 테스트 중...")
        test_embedding = embedding_manager.test_embedding("Instana는 IBM의 애플리케이션 성능 모니터링 솔루션입니다.")
        
        # 4. Milvus 벡터 스토어 초기화
        print("\n4️⃣ Milvus 벡터 스토어 초기화 중...")
        vectorstore_manager = MilvusVectorStoreManager(
            embeddings=embedding_manager.get_embeddings(),
            collection_name="instana_docs"
        )
        
        # 연결 테스트
        print("🔍 Milvus 연결 테스트 중...")
        if not vectorstore_manager.test_connection():
            print("❌ Milvus 연결에 실패했습니다.")
            print("Docker Compose로 Milvus 서버가 실행 중인지 확인해주세요:")
            print("  docker-compose -f milvus-standalone-docker-compose.yml up -d")
            return False
        
        # 5. 문서를 벡터 스토어에 저장
        print("\n5️⃣ 문서를 Milvus에 저장 중...")
        start_time = time.time()
        
        # 배치 처리로 문서 추가 (메모리 효율성을 위해)
        batch_size = 50
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            print(f"   배치 {batch_num}/{total_batches} 처리 중... ({len(batch)}개 문서)")
            
            try:
                vectorstore_manager.add_documents(batch)
                print(f"   ✅ 배치 {batch_num} 완료")
                
            except Exception as e:
                print(f"   ❌ 배치 {batch_num} 실패: {e}")
                return False
        
        end_time = time.time()
        print(f"✅ 모든 문서 저장 완료! (소요 시간: {end_time - start_time:.2f}초)")
        
        # 6. 저장 결과 검증
        print("\n6️⃣ 저장 결과 검증 중...")
        
        # 컬렉션 정보 조회
        collection_info = vectorstore_manager.get_collection_info()
        print(f"📊 컬렉션 정보:")
        print(f"   - 컬렉션명: {collection_info.get('collection_name', 'N/A')}")
        print(f"   - 총 엔티티 수: {collection_info.get('total_entities', 'N/A')}")
        print(f"   - 벡터 차원: {collection_info.get('dimension', 'N/A')}")
        
        # 샘플 검색 테스트
        print("\n🔍 샘플 검색 테스트 중...")
        test_queries = [
            "Instana란 무엇인가요?",
            "애플리케이션 성능 모니터링",
            "IBM의 관찰 가능성 솔루션"
        ]
        
        for query in test_queries:
            print(f"\n   쿼리: '{query}'")
            try:
                results = vectorstore_manager.similarity_search(query, k=3)
                for i, doc in enumerate(results, 1):
                    content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                    print(f"   결과 {i}: {content_preview}")
                    print(f"   메타데이터: {doc.metadata}")
                    
            except Exception as e:
                print(f"   ❌ 검색 실패: {e}")
        
        print("\n" + "=" * 60)
        print("🎉 PDF 문서 저장이 성공적으로 완료되었습니다!")
        print("=" * 60)
        print("\n다음 단계:")
        print("1. Milvus 서버가 실행 중인지 확인")
        print("2. RAG 시스템에서 이 벡터 스토어를 사용하여 검색 기능 구현")
        print("3. Streamlit 앱에서 검색 결과를 활용한 챗봇 응답 생성")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        print("\n문제 해결 방법:")
        print("1. 환경 변수 설정 확인 (.env 파일)")
        print("2. Milvus 서버 실행 상태 확인")
        print("3. PDF 파일 존재 여부 확인")
        print("4. 네트워크 연결 상태 확인")
        return False


def check_prerequisites():
    """사전 요구사항 확인"""
    print("🔍 사전 요구사항 확인 중...")
    
    # PDF 파일 확인
    pdf_path = "data/instana-observability-1.0.301-documentation.pdf"
    if not os.path.exists(pdf_path):
        print(f"❌ PDF 파일이 없습니다: {pdf_path}")
        return False
    
    # 환경 변수 확인
    required_env_vars = ["WATSONX_PROJECT_ID"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ 누락된 환경 변수: {missing_vars}")
        print("다음 환경 변수들을 .env 파일에 설정해주세요:")
        for var in missing_vars:
            print(f"  {var}=your_value_here")
        return False
    
    print("✅ 사전 요구사항 확인 완료")
    return True


if __name__ == "__main__":
    print("Instana PDF to Milvus 벡터 DB 저장 스크립트")
    print("=" * 50)
    
    # 사전 요구사항 확인
    if not check_prerequisites():
        print("\n❌ 사전 요구사항을 충족하지 않습니다.")
        print("위의 문제들을 해결한 후 다시 실행해주세요.")
        sys.exit(1)
    
    # 메인 실행
    success = main()
    
    if success:
        print("\n🎉 스크립트 실행이 성공적으로 완료되었습니다!")
        sys.exit(0)
    else:
        print("\n❌ 스크립트 실행 중 오류가 발생했습니다.")
        sys.exit(1)
