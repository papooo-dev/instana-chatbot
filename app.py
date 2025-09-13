import os
import io
import time
import qrcode
from PIL import Image
from dotenv import load_dotenv

import streamlit as st
from llm import build_chain, build_streaming_chain

# ---------- Config & Session ----------
load_dotenv()

st.set_page_config(page_title="watsonx Chatbot", page_icon="💬", layout="centered")

if "history" not in st.session_state:
    st.session_state.history = []  # list of {"role": "user"|"assistant", "content": str}

if "turns" not in st.session_state:
    st.session_state.turns = 0

if "qr_shown" not in st.session_state:
    st.session_state.qr_shown = False

CHAT_TURNS_LIMIT = int(os.getenv("CHAT_TURNS_LIMIT", "5"))
QR_TEXT = os.getenv("QR_TEXT", "https://example.com/thank-you")

# Build chain once
if "chain" not in st.session_state:
    st.session_state.chain = build_chain()

if "streaming_chain" not in st.session_state:
    st.session_state.streaming_chain = build_streaming_chain()

st.title("💬 watsonx Chatbot")
st.caption("Streamlit + LangChain (watsonx). RAG-ready. uv-managed.")

# ---------- QR Popup Logic ----------
def qr_image_bytes(data: str) -> bytes:
    img = qrcode.make(data)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def show_qr_and_block():
    # If your Streamlit supports modal/dialog, you could do:
    # with st.modal("Chat limit reached"):
    #     st.write("You've reached the maximum number of chat turns.")
    #     st.image(qr_image_bytes(QR_TEXT))
    #     st.stop()

    st.warning("대화 가능 횟수에 도달했습니다. 아래 QR 코드를 스캔해주세요.")
    st.image(qr_image_bytes(QR_TEXT), caption="Scan the QR code", use_container_width=False)
    st.session_state.qr_shown = True


def stream_response_generator(user_input, history):
    """
    스트리밍 응답을 위한 제너레이터 함수
    """
    try:
        # LangChain chain expects {input, history}; history as list of messages
        # Convert to LC-style messages:
        lc_history = []
        for h in history[:-1]:  # exclude current user_input
            if h["role"] == "user":
                lc_history.append(("human", h["content"]))
            else:
                lc_history.append(("ai", h["content"]))

        # 스트리밍 체인 사용
        for chunk in st.session_state.streaming_chain.stream({"input": user_input, "history": lc_history}):
            yield chunk
    except Exception as e:
        yield f"LLM 호출 중 오류가 발생했습니다: {e}"


# ---------- Render Chat History ----------
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- Chat Input (disabled if over limit) ----------
input_disabled = st.session_state.turns >= CHAT_TURNS_LIMIT
user_input = st.chat_input("메시지를 입력하세요...", disabled=input_disabled)

if user_input is not None and not input_disabled:
    # Append user msg
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # LLM respond with streaming
    with st.chat_message("assistant"):
        # 스트리밍 응답 생성
        response_generator = stream_response_generator(user_input, st.session_state.history)
        
        # st.write_stream을 사용하여 스트리밍 출력
        full_response = st.write_stream(response_generator)
        print("::: full_response :::", full_response)

    # Append assistant msg and bump turn count
    st.session_state.history.append({"role": "assistant", "content": str(full_response)})
    st.session_state.turns += 1

    # If reached limit now, show QR
    if st.session_state.turns >= CHAT_TURNS_LIMIT and not st.session_state.qr_shown:
        show_qr_and_block()
        st.stop()

elif input_disabled and not st.session_state.qr_shown:
    # If limit exceeded before render (e.g., after reload), show QR immediately
    show_qr_and_block()
    st.stop()
