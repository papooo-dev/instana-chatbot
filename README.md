# Streamlit + LangChain (watsonx) Chatbot (uv-managed)

A minimal chatbot UI built with **Streamlit** and **LangChain**, using **IBM watsonx** models.
After a configurable number of turns, the app pops up a **QR code** and **blocks further chat**.

## Features
- Streamlit chat UI (`st.chat_message`, `st.chat_input`)
- LangChain with IBM watsonx via `langchain-ibm`
- Ready for future **RAG** with Milvus (stub included)
- `uv` package management
- Turn limit from env (e.g., `CHAT_TURNS_LIMIT=5`) → show QR and stop

## Quickstart

1) **Install uv** (see https://docs.astral.sh/uv/)
```bash
# macOS / Linux (one-liner)
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

2) **Clone / unzip** this project and enter the folder:
```bash
cd streamlit-watsonx-chatbot
```

3) **Copy env and fill in watsonx credentials**:
```bash
cp .env.example .env
# then edit .env with your keys and settings
```

4) **Sync dependencies and run**:
```bash
uv sync
uv run streamlit run app.py
```

### Environment variables
Place these in `.env`:

```ini
# --- IBM watsonx ---
WATSONX_API_KEY=YOUR_API_KEY
WATSONX_URL=https://us-south.ml.cloud.ibm.com  # or your region endpoint
WATSONX_PROJECT_ID=YOUR_PROJECT_GUID
WATSONX_MODEL_ID=ibm/granite-20b-multilingual  # example model

# --- App settings ---
CHAT_TURNS_LIMIT=5
QR_TEXT=https://example.com/thank-you   # encoded into QR after limit reached
SYSTEM_PROMPT=You are a helpful, concise assistant.

# --- (Future) Milvus for RAG ---
MILVUS_URI= # e.g., http://localhost:19530
MILVUS_COLLECTION= # e.g., docs
```

### Notes
- If you prefer `st.modal` for the popup, upgrade Streamlit to a recent version and replace the banner logic with a modal (see comments in `app.py`).

Enjoy! 👋
