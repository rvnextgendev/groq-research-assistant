from unittest.mock import Mock, patch

import pytest
import requests
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake import FakeListLLM
from test_documents import sample_pdf

from research_assistant.agent import ResearchService
from research_assistant.config import Settings
from research_assistant.documents import build_index
from research_assistant.errors import ConfigurationError, ResearchError
from research_assistant.tools import build_tools


def test_credentials_hidden():
    settings = Settings("private-groq", "private-serper")
    assert "private" not in repr(settings)


def test_missing_credentials_rejected():
    with pytest.raises(ConfigurationError):
        Settings("", "")


def test_provider_error_is_sanitized(settings):
    executor = Mock()
    executor.invoke.side_effect = RuntimeError("SECRET_API_KEY")
    with pytest.raises(ResearchError) as exc:
        ResearchService(executor, settings).ask("question", [])
    assert "SECRET" not in str(exc.value)


def test_context_is_bounded(settings):
    executor = Mock()
    executor.invoke.return_value = {"output": "answer"}
    service = ResearchService(executor, settings)
    service.ask("question", [{"role": "user", "content": "x" * 20000}])
    assert len(executor.invoke.call_args.args[0]["chat_history"]) == settings.max_history_chars


@pytest.mark.parametrize("question", ["", "x" * 4001])
def test_invalid_question_does_not_call_model(settings, question):
    executor = Mock()
    with pytest.raises(ResearchError):
        ResearchService(executor, settings).ask(question, [])
    executor.invoke.assert_not_called()


def test_search_failure_is_safe(settings):
    index = build_index(sample_pdf(), settings, DeterministicFakeEmbedding(size=16))
    search = build_tools(index, settings)[1]
    with patch("research_assistant.tools.requests.post", side_effect=requests.Timeout("secret")):
        result = search.invoke("industry growth")
    assert "unavailable" in result
    assert "secret" not in result


def test_real_agent_routes_to_both_tools(settings):
    index = build_index(sample_pdf(), settings, DeterministicFakeEmbedding(size=16))
    llm = FakeListLLM(responses=[
        "Action: pdf_knowledge_base\nAction Input: revenue",
        "Action: web_search\nAction Input: industry growth",
        "Final Answer: PDF growth is 34 percent [PDF page 1]; web: https://example.org/report",
    ])
    response = Mock()
    response.json.return_value = {"organic": [{"title": "Growth", "link": "https://example.org/report"}]}
    with patch("research_assistant.agent.ChatGroq", return_value=llm), patch(
        "research_assistant.tools.requests.post", return_value=response) as request:
        answer = ResearchService.create(index, settings).ask("Compare PDF with web", [])
    assert answer.tools_used == ("pdf_knowledge_base", "web_search")
    assert "PDF page 1" in answer.text
    assert request.call_args.kwargs["timeout"] == (5, 20)
