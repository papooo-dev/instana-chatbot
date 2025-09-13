"""
PDF 문서 처리 모듈
PDF 파일을 텍스트로 변환하고 청크로 분할하는 기능을 제공합니다.
"""
import os
from typing import List, Dict, Any
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class PDFProcessor:
    """PDF 문서를 처리하고 텍스트 청크로 분할하는 클래스"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        PDF 프로세서 초기화
        
        Args:
            chunk_size: 각 청크의 최대 문자 수
            chunk_overlap: 청크 간 겹치는 문자 수
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_pdf(self, pdf_path: str) -> List[Document]:
        """
        PDF 파일을 로드하고 Document 객체 리스트로 반환
        
        Args:
            pdf_path: PDF 파일 경로
            
        Returns:
            Document 객체 리스트
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
        
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # 각 문서에 메타데이터 추가
            for doc in documents:
                doc.metadata.update({
                    "source": pdf_path,
                    "file_name": Path(pdf_path).name,
                    "file_type": "pdf"
                })
            
            return documents
            
        except Exception as e:
            raise Exception(f"PDF 로드 중 오류 발생: {e}")
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        문서들을 청크로 분할
        
        Args:
            documents: 분할할 Document 객체 리스트
            
        Returns:
            분할된 Document 객체 리스트
        """
        try:
            split_docs = self.text_splitter.split_documents(documents)
            
            # 각 청크에 고유 ID 추가
            for i, doc in enumerate(split_docs):
                doc.metadata["chunk_id"] = i
                doc.metadata["total_chunks"] = len(split_docs)
            
            return split_docs
            
        except Exception as e:
            raise Exception(f"문서 분할 중 오류 발생: {e}")
    
    def process_pdf(self, pdf_path: str) -> List[Document]:
        """
        PDF 파일을 로드하고 청크로 분할하는 전체 프로세스
        
        Args:
            pdf_path: PDF 파일 경로
            
        Returns:
            처리된 Document 객체 리스트
        """
        print(f"PDF 파일 로드 중: {pdf_path}")
        documents = self.load_pdf(pdf_path)
        
        print(f"문서 분할 중... (총 {len(documents)} 페이지)")
        split_documents = self.split_documents(documents)
        
        print(f"분할 완료: {len(split_documents)}개 청크 생성")
        return split_documents
    
    def get_document_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """
        문서 통계 정보 반환
        
        Args:
            documents: 분석할 Document 객체 리스트
            
        Returns:
            통계 정보 딕셔너리
        """
        if not documents:
            return {"total_chunks": 0, "total_characters": 0, "avg_chunk_size": 0}
        
        total_chars = sum(len(doc.page_content) for doc in documents)
        avg_chunk_size = total_chars / len(documents)
        
        return {
            "total_chunks": len(documents),
            "total_characters": total_chars,
            "avg_chunk_size": round(avg_chunk_size, 2),
            "min_chunk_size": min(len(doc.page_content) for doc in documents),
            "max_chunk_size": max(len(doc.page_content) for doc in documents)
        }
