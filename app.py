"""
AskStan - Streamlit 애플리케이션
"""
import os
import io
import qrcode
import re
from dotenv import load_dotenv

import streamlit as st
from core.llm import build_streaming_chain, get_rag_context
from traceloop.sdk import Traceloop

# =============================================================================
# 설정 및 세션 상태 초기화
# =============================================================================
load_dotenv()

# =============================================================================
# Instana Generative AI Observability 설정
# =============================================================================
# OTEL_RESOURCE_ATTRIBUTES 환경 변수가 설정되지 않은 경우 기본값 설정
if "OTEL_RESOURCE_ATTRIBUTES" not in os.environ:
    os.environ["OTEL_RESOURCE_ATTRIBUTES"] = "INSTANA_PLUGIN=genai"

# Traceloop SDK 초기화 (Instana 모니터링)
Traceloop.init(app_name="AskStan", disable_batch=True)

st.set_page_config(
    page_title="AskStan", 
    page_icon="data/instana-logo.png", 
    layout="centered"
)

st.markdown("""
<style>
.stButton>button {
    background-color: #006699; /* 녹색 */
    color: white;
    border: 2px solid #006699;
}
.stButton>button:hover {
    background-color: #006699;
    color: white;
    border: 2px solid #006699;
}
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if "history" not in st.session_state:
    st.session_state.history = []  # [{"role": "user"|"assistant", "content": str}]

if "turns" not in st.session_state:
    st.session_state.turns = 0

if "qr_shown" not in st.session_state:
    st.session_state.qr_shown = False

# 환경 변수 설정
CHAT_TURNS_LIMIT = int(os.getenv("CHAT_TURNS_LIMIT"))
QR_TEXT = os.getenv("QR_TEXT")

# 스트리밍 체인 초기화 (RAG 통합)
if "streaming_chain" not in st.session_state:
    st.session_state.streaming_chain = build_streaming_chain()


def qr_image_bytes(data: str) -> bytes:
    """QR 코드 이미지를 바이트로 변환"""
    img = qrcode.make(data)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@st.dialog("데모 참여 완료!🎉", width="small", dismissible=False)
def show_qr_dialog():
    """대화 한도 도달 시 QR 코드를 팝업으로 표시"""
    st.markdown("아래 QR 코드를 스마트폰을 통해 스캔하여 설문에 참여해주세요!")
    st.markdown("설문 완료 시, IBM Quantum 티셔츠를 드립니다. 🤗")
    
    # QR 코드 이미지 표시
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(qr_image_bytes(QR_TEXT), width=300)
    
    st.markdown("---")
    
    # 대화 다시 시작하기 버튼
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("처음으로 돌아가기", type="primary", use_container_width=True):
            # 세션 상태 초기화
            st.session_state.history = []
            st.session_state.turns = 0
            st.session_state.qr_shown = False
            st.rerun()




def stream_response_generator(user_input: str, history: list):
    """RAG 기반 스트리밍 응답을 위한 제너레이터 함수"""
    try:
        # LangChain 형식으로 히스토리 변환
        lc_history = []
        for h in history[:-1]:  # 현재 사용자 입력 제외
            if h["role"] == "user":
                lc_history.append(("human", h["content"]))
            else:
                lc_history.append(("ai", h["content"]))

        # RAG 통합 체인으로 스트리밍 응답 처리
        for chunk in st.session_state.streaming_chain.stream({
            "input": user_input, 
            "history": lc_history
        }):
            yield chunk
                    
    except Exception as e:
        yield f"RAG LLM 호출 중 오류가 발생했습니다: {e}"


# =============================================================================
# UI 렌더링
# =============================================================================
# 헤더 섹션
col1, col2 = st.columns([1, 7])
with col1:
    st.image("data/instana-logo.png", width=80)
with col2:
    st.title("AskStan")

st.caption("IBM의 Instana에 대해 궁금한 질문을 물어보세요!")

# 대화 히스토리 렌더링
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 대화 한도 도달 시 설문 버튼 표시
if st.session_state.turns >= CHAT_TURNS_LIMIT and not st.session_state.qr_shown:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🎁 설문 후 선물받기", type="primary", use_container_width=True):
            show_qr_dialog()
            st.session_state.qr_shown = True

# =============================================================================
# 채팅 입력 및 처리
# =============================================================================
input_disabled = st.session_state.turns >= CHAT_TURNS_LIMIT
user_input = st.chat_input("메시지를 입력하세요...", disabled=input_disabled)

if user_input is not None and not input_disabled:
    # 사용자 메시지 추가 및 표시
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # RAG 기반 응답 생성
    with st.chat_message("assistant"):
        # RAG 모드: 문서 검색 후 응답 생성
        with st.spinner("📚 관련 문서를 검색하고 있습니다..."):
            # RAG 컨텍스트 미리 생성 (사용자에게 피드백 제공)
            rag_context = get_rag_context(user_input)
            
        
        # RAG 기반 스트리밍 응답
        response_generator = stream_response_generator(user_input, st.session_state.history)
        full_response = st.write_stream(response_generator)

    # 어시스턴트 메시지 히스토리에 추가 및 턴 카운트 증가
    st.session_state.history.append({"role": "assistant", "content": full_response})
    st.session_state.turns += 1

    # 대화 한도 도달 시 페이지 새로고침하여 버튼 표시
    if st.session_state.turns >= CHAT_TURNS_LIMIT:
        st.rerun()