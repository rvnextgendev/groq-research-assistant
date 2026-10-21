"""Exercise the compiled graph with scripted structured model responses."""
from time import monotonic
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from research_assistant.errors import ResearchError
from research_assistant.langgraph_agent import LangGraphResearchService, build_graph


def tool_call(name, identifier):
    return AIMessage(content="", tool_calls=[{"name": name, "args": {"query": "revenue"}, "id": identifier}])


@pytest.fixture
def tools():
    @tool
    def pdf_knowledge_base(query: str) -> str:
        """Search the PDF."""
        return "Revenue grew 34 percent [PDF page 1]"

    @tool
    def web_search(query: str) -> str:
        """Search the web."""
        return "Industry data: https://example.org/report"

    return [pdf_knowledge_base, web_search]


def service(settings, tools, responses):
    model = Mock()
    model.bind_tools.return_value.invoke.side_effect = responses
    return LangGraphResearchService(build_graph(model, tools), settings), model


@pytest.mark.parametrize("names", [["pdf_knowledge_base"], ["web_search"],
                                  ["pdf_knowledge_base", "web_search"]])
def test_graph_routes_tools_and_finishes(settings, tools, names):
    responses = [tool_call(name, str(i)) for i, name in enumerate(names)]
    agent, model = service(settings, tools, [*responses, AIMessage(content="Sourced answer")])
    answer = agent.ask("Find evidence", [])
    assert answer.text == "Sourced answer"
    assert answer.tools_used == tuple(names)
    # The graph must return tool observations to the model before its final answer.
    final_messages = model.bind_tools.return_value.invoke.call_args.args[0]
    observations = [message for message in final_messages if isinstance(message, ToolMessage)]
    assert [m.tool_call_id for m in observations] == [str(i) for i in range(len(names))]


def test_graph_stops_repeated_tool_requests(settings, tools):
    agent, model = service(settings, tools, [tool_call("web_search", str(i)) for i in range(10)])
    with pytest.raises(ResearchError, match="limit"):
        agent.ask("Keep searching", [])
    assert model.bind_tools.return_value.invoke.call_count == 9


def test_graph_sanitizes_provider_error(settings, tools):
    agent, _ = service(settings, tools, [RuntimeError("SECRET_API_KEY")])
    with pytest.raises(ResearchError) as exc:
        agent.ask("question", [])
    assert "SECRET" not in str(exc.value)


def test_graph_sanitizes_tool_error_and_can_recover(settings):
    @tool
    def failing_tool(query: str) -> str:
        """A tool unavailable during this test."""
        raise RuntimeError("PRIVATE_DOCUMENT_TEXT")

    agent, model = service(settings, [failing_tool],
        [tool_call("failing_tool", "1"), AIMessage(content="Source unavailable")])
    assert agent.ask("question", []).text == "Source unavailable"
    observation = model.bind_tools.return_value.invoke.call_args.args[0][-1]
    assert "PRIVATE" not in observation.content
    assert "unavailable" in observation.content


def test_graph_bounds_history_and_validates_questions(settings):
    graph = Mock()
    graph.invoke.return_value = {"messages": [AIMessage(content="answer")]}
    agent = LangGraphResearchService(graph, settings)
    agent.ask("question", [{"role": "user", "content": "x" * 20000}])
    context = graph.invoke.call_args.args[0]["messages"][0].content
    assert context.endswith("x" * settings.max_history_chars)
    assert len(context) < settings.max_history_chars + 100
    graph.reset_mock()
    for question in ("", "x" * 4001):
        with pytest.raises(ResearchError):
            agent.ask(question, [])
    graph.invoke.assert_not_called()


def test_graph_deadline_prevents_model_call(settings, tools):
    agent, model = service(settings, tools, [])
    with pytest.raises(ResearchError, match="time limit"):
        agent.graph.invoke({"messages": [], "rounds": 0, "deadline": monotonic() - 1})
    model.bind_tools.return_value.invoke.assert_not_called()
