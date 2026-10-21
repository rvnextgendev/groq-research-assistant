# Groq Research Assistant

A modular Streamlit application combining PDF retrieval (local embeddings and FAISS) with Groq and Serper web search. The agent can use either source or both. Answers include requested page/URL citations, tool names, follow-up context, and a conversation download.

## Run locally

Python 3.11 is the tested runtime.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
# Enter fresh GROQ_API_KEY and SERPER_API_KEY in .env
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open http://localhost:8501, upload a PDF, click **Process PDF**, and ask a question. `streamlit run main.py` also works. The previous terminal interaction is replaced by the UI.

Examples: "What does this document say about revenue?"; "Compare this with current industry growth."

## Structure

```text
app.py                         Streamlit deployment entry point
research_assistant/
  config.py                    Validated environment/secrets configuration
  documents.py                 PDF parsing, chunking, embedding, FAISS
  tools.py                     PDF retrieval and bounded Serper requests
  prompts.py                   Research instructions and citation format
  agent.py                     ReAct service, history and error boundaries
  errors.py                    Safe application exceptions
  ui.py                        Uploads, session state, chat and downloads
tests/                         Offline service and Streamlit UI tests
.streamlit/                    UI/server configuration and secrets template
Dockerfile                     Non-root container with health check
.github/workflows/ci.yml        Lint and tests
```

## Configuration and data handling

Set `GROQ_API_KEY`, `SERPER_API_KEY`, and optionally `LLM_MODEL` in `.env`. Alternatively copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`. Streamlit secrets take precedence over environment variables. Credentials never appear in configuration repr or application error logs. Use fresh keys; never commit secrets.

Uploads are limited to 10 MB, 200 pages, and one million extracted characters. Scanned PDFs require OCR; encrypted files are rejected. PDF parsing and indexing run in the app process. Embeddings download on first use and require internet access and sufficient memory. Relevant extracts are sent to Groq; web queries go to Serper. PDF bytes are processed in memory, not written to shared files. Indexes and chat are held in per-session state, without a global document cache. Browser refresh or server restart can reset the session. Downloads are generated per user rather than written to a shared answer.txt.

Agent requests use eight iterations, a cooperative 120-second execution limit, finite provider timeouts and retries, and bounded chat history. The execution limit is checked between steps; it is not a hard process deadline. Tool activity labels are shown instead of raw internal reasoning. Citations are model-generated and should be reviewed against source material.

## Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository, excluding `.env`, private PDFs, and `secrets.toml`.
2. Create an app in Streamlit Community Cloud, select your repository/branch and `app.py` as entry point. Select Python 3.11 in advanced settings.
3. Paste the contents of `.streamlit/secrets.toml.example` into deployment Secrets and replace placeholders with fresh keys.
4. Deploy. The platform installs `requirements.txt`; initial embedding download happens when the first PDF is processed.

See the official [deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) and [secrets guide](https://docs.streamlit.io/deploy/concepts/secrets).

## Docker

```sh
docker build -t research-assistant .
docker run --rm -p 8501:8501 --env-file .env research-assistant
```

The container runs as a non-root user and exposes a Streamlit health check. Docker startup has not been validated unless explicitly noted in the delivery report.

## Validate

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q
```

Tests use generated PDFs and fake models/embeddings, with no live API calls or model downloads. `requirements-lock.txt` records the tested runtime dependency versions; use `pip install -r requirements-lock.txt` to reproduce them.

## Deployment scope

This is a deployable, tested application foundation, not a claim of production certification. It has no built-in authentication, durable storage, per-user billing/quota enforcement, or distributed job workers. Before exposing paid server keys publicly, restrict access at the hosting layer or add authentication and rate limiting. Hostile PDF parsing needs a separately sandboxed worker for untrusted public uploads; page/text limits do not guarantee CPU or decompression bounds. Load-test against expected concurrent users and hosting memory limits before a production rollout. Live Groq/Serper credentials and the target deployment must be verified separately.
