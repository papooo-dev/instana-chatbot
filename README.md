# IBM Instana 챗봇

**Streamlit**과 **LangChain**으로 구축된 종합적인 챗봇 애플리케이션으로, **Mistral AI** 모델을 기반으로 하며 **Milvus** 벡터 데이터베이스를 활용한 **RAG(검색 증강 생성)** 기능을 제공합니다.

## 🌟 주요 기능

- **대화형 채팅 UI**: 스트리밍 응답을 지원하는 Streamlit 기반 채팅 인터페이스
- **RAG 기능**: Milvus 벡터 데이터베이스를 활용한 문서 검색 및 컨텍스트 인식 응답. ([활용 문서](https://www.ibm.com/docs/en/SSE1JP5_1.0.301/pdf/instana-observability-1.0.301-documentation.pdf))
- **uv 패키지 관리**: Python 의존성 관리

## 🚀 Quick Start
 
### 사전 요구사항

1. **uv 설치** (https://docs.astral.sh/uv/ 참조)
```bash
# macOS / Linux (원라이너)
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

2. **프로젝트 클론**:
```bash
git clone https://github.com/papooo-dev/instana-chatbot.git
cd instana-chatbot
```

### 환경 설정

1. **환경 파일 생성**:
```bash
cp .env.example .env
```

2. **환경 변수 설정** (`.env` 파일에):
```ini
# --- IBM watsonx ---
WATSONX_API_KEY=YOUR_API_KEY
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WATSONX_PROJECT_ID=YOUR_PROJECT_GUID
WATSONX_MODEL_ID=ibm/granite-20b-multilingual

# --- 앱 설정 ---
CHAT_TURNS_LIMIT=CHAT_LIMIT
QR_TEXT=YOUR_QR_CODE_URL

# --- RAG용 Milvus ---
MILVUS_URI=http://localhost:19530
MILVUS_COLLECTION=instana_docs
```

### 설치 및 실행

1. **의존성 설치**:
```bash
uv sync
```

2. **PDF 문서 다운로드** (RAG 기능용):
```bash
# Instana 공식 문서 다운로드
curl -o data/instana-observability-1.0.301-documentation.pdf \
  "https://www.ibm.com/docs/en/SSE1JP5_1.0.301/pdf/instana-observability-1.0.301-documentation.pdf"
```

3. **Milvus 서버 시작** (RAG 기능용):
```bash
# Docker Compose 사용
docker-compose -f milvus-standalone-docker-compose.yml up -d

# 서버 상태 확인
docker-compose -f milvus-standalone-docker-compose.yml ps
```

4. **PDF 문서 벡터화** (RAG 기능용):
```bash
# PDF 문서 저장 스크립트 실행
uv run utils/ingest_pdf_to_milvus.py
```

5. **애플리케이션 실행**:
```bash
uv run streamlit run app.py
```

## 📚 프로젝트 구조

```
instana-chatbot/
├── app.py                 # 메인 Streamlit 애플리케이션
├── core/                  # 핵심 기능 모듈
│   ├── llm.py            # LLM 체인 및 스트리밍 설정
│   ├── rag.py            # RAG 구현
│   ├── milvus_manager.py # Milvus 벡터 데이터베이스 관리
│   ├── embedding.py      # Watsonx 임베딩 통합
│   └── prompts.py        # 시스템 프롬프트 및 템플릿
├── utils/                 # 유틸리티 함수
├── data/                  # 정적 자산 및 문서
├── config/                # 설정 파일
└── volumes/               # Milvus 데이터 저장소
```

## 🔧 핵심 구성 요소

### 채팅 인터페이스 (`app.py`)
- `st.chat_message`와 `st.chat_input`을 사용한 Streamlit 기반 채팅 UI
- 턴 기반 대화 관리
- 턴 제한 후 QR 코드 표시
- 컨텍스트 인식 응답을 위한 RAG 통합

### LLM 통합 (`core/llm.py`)
- `langchain-ibm`을 통한 IBM watsonx와의 LangChain 통합
- 스트리밍 응답 지원
- LLM을 위한 RAG 컨텍스트 통합

### RAG 시스템 (`core/rag.py`)
- 문서 검색 및 순위 매기기
- LLM을 위한 컨텍스트 조립
- Milvus 벡터 검색 통합

### 벡터 데이터베이스 (`core/milvus_manager.py`)
- Milvus 컬렉션 관리
- 문서 저장 및 검색
- 유사도 검색 기능

## 📖 PDF to Milvus 벡터 DB 저장 가이드

이 가이드는 Instana PDF 문서를 Milvus 벡터 데이터베이스에 저장하는 방법을 설명합니다.

### 📋 사전 요구사항

#### 1. 필요한 패키지 설치
```bash
# uv를 사용한 패키지 설치
uv sync

# 또는 pip 사용
pip install -r requirements.txt
```

### 🚀 실행 방법

#### 1. Milvus 서버 시작
```bash
# Docker Compose로 Milvus 서버 시작
docker-compose -f config/milvus-standalone-docker-compose.yml up -d

# 서버 상태 확인
docker-compose -f config/milvus-standalone-docker-compose.yml ps
```

#### 2. PDF 문서 다운로드 및 저장
RAG 기능을 위해 Instana 문서를 다운로드하여 `data/` 디렉토리에 저장해야 합니다:

```bash
# PDF 문서 다운로드 (IBM 공식 문서)
curl -o data/instana-observability-1.0.303-documentation.pdf \
  "https://www.ibm.com/docs/en/SSE1JP5_1.0.303/pdf/instana-observability-1.0.303-documentation.pdf"
```

또는 브라우저에서 직접 다운로드:
- [Instana Observability 1.0.303 Documentation PDF](https://www.ibm.com/docs/en/SSE1JP5_1.0.303/pdf/instana-observability-1.0.303-documentation.pdf)
- 다운로드한 파일을 `data/` 디렉토리에 저장

#### 3. PDF 문서 저장 스크립트 실행
```bash
# 메인 스크립트 실행
uv run utils/ingest_pdf_to_milvus.py
```

### 🔧 주요 기능

#### PDF 처리 (`pdf_processor.py`)
- PDF 파일을 텍스트로 변환
- 텍스트를 청크로 분할 (기본: 1000자, 200자 겹침)
- 메타데이터 추가 (파일명, 청크 ID 등)

#### Milvus 벡터 스토어 (`milvus_manager.py`)
- LangChain Milvus 통합
- 문서 저장 및 검색
- 유사도 검색 기능

## 🛠️ 개발

### 새 문서 추가
1. `data/` 디렉토리에 PDF 파일 배치
2. 수집 스크립트를 실행하여 문서 처리 및 저장
3. 챗봇이 자동으로 새 문서를 RAG에 활용

### 프롬프트 사용자 정의
`core/prompts.py`를 편집하여 시스템 프롬프트와 응답 템플릿을 수정할 수 있습니다.

### 기능 확장
- `utils/`에 새 문서 프로세서 추가
- `core/rag.py`에서 RAG 기능 확장
- `app.py`에서 채팅 인터페이스 수정

## 📝 참고사항
- RAG 기능을 사용하려면 실행 중인 Milvus 인스턴스가 필요합니다.

---

즐거운 사용 되세요! 👋