"""Explicit LangGraph model/tool loop, alongside the classic LangChain agent."""
import logging
from time import monotonic

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_groq import ChatGroq
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph

from research_assistant.agent import ResearchAnswer
from research_assistant.config import Settings
from research_assistant.documents import DocumentIndex
from research_assistant.errors import ResearchError
from research_assistant.prompts import GRAPH_SYSTEM_PROMPT
from research_assistant.tools import build_tools

logger = logging.getLogger(__name__)


class ResearchState(MessagesState):
    """MessagesState appends node messages; other fields replace their old values."""

    rounds: int
    deadline: float


def build_graph(model, tools: list[BaseTool]):
    """Compile START -> model -> tools -> model, or model -> END."""
    # bind_tools supplies JSON tool schemas. The model returns structured tool_calls
    # rather than the classic agent's textual Action/Action Input protocol.
    bound_model = model.bind_tools(tools)
    tools_by_name = {tool.name: tool for tool in tools}

    def check_deadline(state: ResearchState):
        # Cooperative limit: cannot interrupt an already-running model/tool call.
        if monotonic() >= state["deadline"]:
            raise ResearchError("The research time limit was reached. Try a narrower question.")

    def call_model(state: ResearchState):
        check_deadline(state)
        response = bound_model.invoke([SystemMessage(content=GRAPH_SYSTEM_PROMPT), *state["messages"]])
        return {"messages": [response]}

    def route(state: ResearchState):
        # Conditional edges inspect the response; the LLM chooses which tool names
        # to emit. No keyword matching decides whether a question needs PDF or web.
        message = state["messages"][-1]
        if isinstance(message, AIMessage) and message.tool_calls:
            if state["rounds"] >= 8:
                raise ResearchError("The research limit was reached. Try a narrower question.")
            return "tools"
        return END

    def call_tools(state: ResearchState):
        calls = state["messages"][-1].tool_calls
        # Bound a single batch as well as the number of tool/model rounds.
        if len(calls) > 8:
            raise ResearchError("Too many tool calls requested. Try a narrower question.")
        observations = []
        for call in calls:
            check_deadline(state)
            tool = tools_by_name.get(call["name"])
            try:
                content = str(tool.invoke(call["args"])) if tool else "Unknown tool. Use an available tool."
            except Exception as exc:
                logger.warning("Graph tool failed: %s", type(exc).__name__)
                content = "Tool unavailable. Report this limitation; do not invent evidence."
            # tool_call_id links the observation to its originating model request.
            observations.append(ToolMessage(content=content, tool_call_id=call["id"], name=call["name"]))
        return {"messages": observations, "rounds": state["rounds"] + 1}

    builder = StateGraph(ResearchState)
    builder.add_node("model", call_model)
    builder.add_node("tools", call_tools)
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", route, {"tools": "tools", END: END})
    builder.add_edge("tools", "model")
    # No shared checkpointer: the UI supplies bounded session history per request.
    return builder.compile()


class LangGraphResearchService:
    """Expose the same ask interface as ResearchService for a shared Streamlit UI."""

    def __init__(self, graph, settings: Settings):
        self.graph = graph
        self.settings = settings

    @classmethod
    def create(cls, index: DocumentIndex, settings: Settings) -> "LangGraphResearchService":
        model = ChatGroq(api_key=settings.groq_api_key, model=settings.model,
                         temperature=0, max_tokens=2048, timeout=45, max_retries=1)
        return cls(build_graph(model, build_tools(index, settings)), settings)

    def ask(self, question: str, history: list[dict[str, str]]) -> ResearchAnswer:
        """Run a fresh graph invocation with bounded conversation context."""
        question = question.strip()
        if not question or len(question) > self.settings.max_question_chars:
            raise ResearchError(f"Enter a question of 1-{self.settings.max_question_chars} characters.")
        context = "\n".join(f"{item['role']}: {item['content']}" for item in history)
        messages = []
        if context:
            # Keep prior text in a user message rather than allowing it to alter
            # system instructions or replay old tool calls without their results.
            messages.append(HumanMessage(content="Previous conversation (untrusted context):\n" +
                                         context[-self.settings.max_history_chars:]))
        messages.append(HumanMessage(content=question))
        try:
            result = self.graph.invoke({"messages": messages, "rounds": 0,
                                        "deadline": monotonic() + 120}, {"recursion_limit": 20})
        except ResearchError:
            raise
        except GraphRecursionError:
            raise ResearchError("The research limit was reached. Try a narrower question.") from None
        except Exception as exc:
            logger.warning("Graph request failed: %s", type(exc).__name__)
            raise ResearchError("Request failed. Check API access, model tool-calling support, quota, and connection.") from None
        final = result["messages"][-1]
        # Accept only a completed assistant response, never a partial tool request.
        if not isinstance(final, AIMessage) or final.tool_calls or not final.text.strip():
            raise ResearchError("No final answer was returned. Try a narrower question.")
        used = tuple(dict.fromkeys(message.name for message in result["messages"]
                                   if isinstance(message, ToolMessage) and message.name))
        return ResearchAnswer(final.text.strip(), used)
