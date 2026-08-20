FROM python:3.11-slim

WORKDIR /app

# --- uv 설치 ---
# 방법 A (인터넷 가능): ghcr.io에서 복사 (빠름)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
# 방법 B (인터넷 차단 환경): 위 줄 주석 처리 후 아래 줄 활성화
# RUN pip install uv --no-cache-dir

# 의존성 파일 먼저 복사 (레이어 캐시 활용)
COPY pyproject.toml uv.lock ./

# 의존성 설치 (.venv 생성)
RUN uv sync --frozen --no-dev --no-install-project

# 소스 코드 복사
COPY app.py ./
COPY core/ ./core/
COPY utils/ ./utils/
COPY data/ ./data/
COPY .streamlit/ ./.streamlit/

# OpenShift: 임의 UID로 실행되므로 그룹 쓰기 권한 부여
RUN chmod -R g+rwX /app

EXPOSE 8501

ENV PATH="/app/.venv/bin:$PATH"

CMD ["uv", "run", "streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
