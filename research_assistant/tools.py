"""Read-only tools with explicit request timeouts."""
import json

import requests
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool, create_retriever_tool

from research_assistant.config import Settings
from research_assistant.documents import DocumentIndex


def build_tools(index: DocumentIndex, settings: Settings) -> list:
    """Expose only the current PDF retriever and a fixed-endpoint web search tool."""
    # Tool descriptions help the model decide which source matches a question.
    # Format every retrieved chunk with its source page before passing it to Groq.
    pdf = create_retriever_tool(index.retriever, "pdf_knowledge_base",
        "Search the uploaded PDF for document-specific questions.",
        document_prompt=PromptTemplate.from_template("[PDF page {page_number}] {page_content}"))

    def search_web(query: str) -> str:
        """Return a compact set of search snippets and URLs, not full web pages."""
        if not query.strip() or len(query) > 1000:
            return "Search query must contain 1-1000 characters."
        try:
            # Pass the key in a header; timeout values are connect/read seconds.
            response = requests.post("https://google.serper.dev/search",
                headers={"X-API-KEY": settings.serper_api_key},
                json={"q": query, "num": 3}, timeout=(5, 20))
            response.raise_for_status()
            results = response.json().get("organic", [])[:3]
            # Limit fields and field lengths to bound the tool observation size.
            return json.dumps([{key: str(item.get(key, ""))[:2000]
                                for key in ("title", "link", "snippet")}
                               for item in results])
        except (requests.RequestException, ValueError, AttributeError, TypeError):
            # A safe observation lets the agent explain failure or use PDF evidence.
            return "Web search unavailable. Do not invent web findings; report this limitation."

    return [pdf, Tool(name="web_search", func=search_web,
                      description="Search current web facts; results include source URLs.")]
