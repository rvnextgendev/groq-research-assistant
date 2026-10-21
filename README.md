# Groq Research Assistant

A modular Streamlit application combining PDF retrieval (local embeddings and FAISS) with Groq and Serper web search. The agent can use either source or both. Answers include requested page/URL citations, tool names, follow-up context, and a conversation download.

New to the code? Read the [beginner's implementation guide](docs/IMPLEMENTATION_GUIDE.md) for step-by-step LangChain and LangGraph explanations, `agent_scratchpad` versus message state, and easy PDF/web examples.

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

## Try the included sample PDF

Use [research_assistant_test.pdf](sample_documents/research_assistant_test.pdf), a three-page report with fictional company data. On GitHub, open the PDF and use **Download raw file** to save it to your computer. Do not treat its figures as real company or industry data.

1. Open the local app at http://localhost:8501 or your deployed Streamlit app. Configure Groq and Serper keys first; see the configuration and deployment sections below.
2. In the sidebar, upload `research_assistant_test.pdf` and click **Process PDF**. The first run may take longer while the embedding model downloads.
3. Wait for **Document ready**, then enter the questions below in the chat box.
4. Check page citations and the **Sources searched** label beneath each answer.
5. Ask a follow-up question, use **Download conversation** to export the chat, or **Clear chat** to start a new conversation about the same PDF.

| What to test | Example question | Expected evidence or behavior |
|---|---|---|
| PDF retrieval | What was revenue growth in 2025? Cite the PDF page. | 34 percent; page 1; `pdf_knowledge_base`. |
| PDF retrieval | Explain the RAG workflow described in this report. | PDF extraction, 1,000-character chunks with 200-character overlap, local embeddings, FAISS, up to four retrieved chunks, Groq and Serper; page 2. |
| Target versus result | What is the 2026 revenue target? Has it been achieved? | USD 16 million is a target; the report does not establish that it was achieved; page 3. |
| Missing evidence | What was the company's net profit? | The report does not provide net profit; page 3. The agent should not invent a number. |
| Web search | Search the web for the latest LangChain release and include source URLs. | `web_search`; an answer based on live search results with URLs. Results change over time. |
| PDF and web | Compare the RAG design in this PDF with current guidance in official LangChain documentation. Search both sources. | Both `pdf_knowledge_base` and `web_search`, with PDF page citations and web URLs. |
| Follow-up context | Explain your previous answer in simpler terms. | An explanation that refers to the preceding exchange. |

### How the agent chooses tools

Groq receives your question, recent conversation, tool descriptions, and the research prompt. The prompt directs it to search the PDF for document-specific questions, the web for current facts, and both for comparisons. The ReAct executor runs the selected tool, returns its results to the model, and repeats until the model answers or reaches a configured limit. Routing is model-driven and may vary; **Sources searched** reports the tools actually called, rather than guaranteeing which tool a question will select.

The fictional Northstar Analytics company is not intended to be found online. Use web search to research its technologies or compare the described architecture, rather than trying to verify its invented financial figures.

## Structure

```text
app.py                         Streamlit deployment entry point
research_assistant/
  config.py                    Validated environment/secrets configuration
  documents.py                 PDF parsing, chunking, embedding, FAISS
  tools.py                     PDF retrieval and bounded Serper requests
  prompts.py                   Research instructions and citation format
  agent.py                     ReAct service, history and error boundaries
  langgraph_agent.py            Explicit StateGraph model/tool loop
  errors.py                    Safe application exceptions
  ui.py                        Uploads, session state, chat and downloads
tests/                         Offline service and Streamlit UI tests
sample_documents/              Fictional PDF for manual UI testing
.streamlit/                    UI/server configuration and secrets template
Dockerfile                     Non-root container with health check
.github/workflows/ci.yml        Lint and tests
```

## Compare LangChain and LangGraph

Both implementations are included and share the same PDF indexing, Serper tool, Groq configuration, and chat UI. The existing `agent.py` remains the default.

1. Install the updated dependencies: `python -m pip install -r requirements.txt`.
2. Start the app: `python -m streamlit run app.py`.
3. Select **LangChain ReAct** or **LangGraph** under **Agent implementation** in the sidebar.
4. Upload the sample PDF, select **Process PDF**, and try the example questions above.
5. To compare the other engine, change the selector and click **Process PDF** again. Switching engines clears the old service and conversation; the selected upload remains available.

| Detail | LangChain ReAct (`agent.py`) | LangGraph (`langgraph_agent.py`) |
|---|---|---|
| Model decisions | Textual `Action` / `Action Input` parsed by the executor | Structured `AIMessage.tool_calls` from `bind_tools` |
| Execution loop | `create_react_agent` and `AgentExecutor` | Explicit `StateGraph` model and tools nodes |
| Tool result context | `agent_scratchpad` in the prompt | `ToolMessage` entries appended to `MessagesState` |
| Stop condition | Final answer or executor limit | No tool calls, round limit, deadline, or graph recursion limit |
| Session context | Bounded history supplied by UI | Bounded history supplied by UI; no shared checkpointer |

The LangGraph path is `START -> model -> tools -> model`, repeating as needed, then `model -> END` when the response has no tool calls. Conditional edges inspect the model response; the model still chooses PDF search, web search, or both. The graph has an eight-tool-round limit, at most eight calls per round, a cooperative 120-second deadline, and a 20-step recursion ceiling. Provider calls have their own timeouts. Use a Groq model that supports structured tool calling for this backend. The classic prompt is not reused: `GRAPH_SYSTEM_PROMPT` replaces the textual ReAct format.

Both backends return the same answer/tool-label structure. Offline tests verify graph routing and termination; a live comparison requires your API keys. See the official [LangGraph quickstart](https://docs.langchain.com/oss/python/langgraph/quickstart) for graph concepts.

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
