"""Streamlit UI; document indexes are private to each session."""
import logging

import streamlit as st

from research_assistant.agent import ResearchService
from research_assistant.config import Settings
from research_assistant.documents import build_index
from research_assistant.errors import ResearchError

logger = logging.getLogger(__name__)


def clear_document():
    """Discard the active index and related chat when the uploaded file changes."""
    for key in ("service", "document_stats", "messages"):
        st.session_state.pop(key, None)


def run():
    """Render the app; Streamlit reruns this function on widget interactions."""
    st.set_page_config(page_title="Research Assistant", page_icon=":books:", layout="wide")
    st.title("Research Assistant")
    st.caption("Ask your PDF, explore the web, or compare both - powered by Groq.")
    try:
        try:
            secrets = dict(st.secrets)
        except FileNotFoundError:
            # Local use needs only .env; a Streamlit secrets file is optional.
            secrets = {}
        settings = Settings.load(secrets)
    except ResearchError as exc:
        st.error(str(exc))
        st.info("Set API keys in .env locally or in Streamlit deployment secrets.")
        return

    with st.sidebar:
        st.header("Your document")
        uploaded = st.file_uploader("Upload a PDF", type=["pdf"], on_change=clear_document)
        st.caption(f"Up to {settings.max_upload_mb} MB and {settings.max_pages} pages. Text PDFs only.")
        st.caption("Relevant PDF extracts go to Groq. Web queries go to Serper.")
        st.caption("Documents and chat stay in this session's server memory; refresh may reset them.")
        st.text(f"Model: {settings.model}")
        process = st.button("Process PDF", disabled=uploaded is None, type="primary")
        if st.button("Clear chat"):
            st.session_state["messages"] = []

    if process and uploaded is not None:
        # Discard the old index first so failed processing cannot use stale evidence.
        clear_document()
        try:
            with st.spinner("Reading PDF and building its search index..."):
                index = build_index(uploaded.getvalue(), settings)
                service = ResearchService.create(index, settings)
            # Publish the service only after indexing and agent construction succeed.
            # Session state avoids rebuilding on reruns without sharing users' PDFs.
            st.session_state.update(service=service,
                document_stats=f"{index.page_count} pages / {index.chunk_count} chunks", messages=[])
        except ResearchError as exc:
            st.error(str(exc))
        except Exception as exc:
            # Avoid logging document contents or provider exception messages.
            logger.warning("Index creation failed: %s", type(exc).__name__)
            st.error("Could not prepare the document. Check embedding download, model configuration, and memory.")

    service = st.session_state.get("service")
    if service is None:
        st.info("Upload a PDF and select Process PDF to start researching.")
    else:
        st.success(f"Document ready: {st.session_state['document_stats']}")
    messages = st.session_state.setdefault("messages", [])
    # Redraw completed exchanges from this session on every rerun.
    for item in messages:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item.get("tools"):
                st.caption("Sources searched: " + item["tools"])

    if question := st.chat_input("Ask about your PDF or current web information", disabled=service is None,
                                  max_chars=settings.max_question_chars):
        with st.chat_message("user"):
            st.markdown(question)
        try:
            with st.spinner("Researching..."):
                answer = service.ask(question, messages)
            # Store only successful exchanges, leaving failed questions retryable.
            messages.extend([{"role": "user", "content": question},
                {"role": "assistant", "content": answer.text, "tools": ", ".join(answer.tools_used)}])
            # Bound server-side history as well as the context sent to the model.
            del messages[:-40]
            # Re-render from saved state so the transcript and tool labels stay aligned.
            st.rerun()
        except ResearchError as exc:
            st.error(str(exc))

    if messages:
        # Generate each user's download in memory rather than a shared server file.
        transcript = "\n\n".join(f"{m['role'].title()}: {m['content']}" for m in messages)
        st.download_button("Download conversation", transcript, "research.txt", "text/plain")
