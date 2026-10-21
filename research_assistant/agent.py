"""UI-independent service with bounded history and sanitized errors."""
import logging
from dataclasses import dataclass

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_groq import ChatGroq

from research_assistant.config import Settings
from research_assistant.documents import DocumentIndex
from research_assistant.errors import ResearchError
from research_assistant.prompts import RESEARCH_PROMPT
from research_assistant.tools import build_tools

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResearchAnswer:
    """Final response and tool names for display, without exposing raw agent traces."""
    text: str
    tools_used: tuple[str, ...]


class ResearchService:
    """Keep model execution independent of Streamlit so it can be tested offline."""
    def __init__(self, executor: AgentExecutor, settings: Settings):
        self.executor = executor
        self.settings = settings

    @classmethod
    def create(cls, index: DocumentIndex, settings: Settings) -> "ResearchService":
        """Bind the session's document tools to a Groq-backed ReAct executor."""
        # Provider timeouts bound individual calls; low temperature reduces variation.
        llm = ChatGroq(api_key=settings.groq_api_key, model=settings.model,
                       temperature=0, max_tokens=2048, timeout=45, max_retries=1)
        tools = build_tools(index, settings)
        # The model selects a tool, observes its result, and can select another tool.
        agent = create_react_agent(llm, tools, RESEARCH_PROMPT)
        # The execution deadline is cooperative: checked between agent steps.
        # Keep intermediate steps for tool labels, but do not print private tool data.
        executor = AgentExecutor(agent=agent, tools=tools, verbose=False,
            max_iterations=8, max_execution_time=120, handle_parsing_errors=True,
            return_intermediate_steps=True)
        return cls(executor, settings)

    def ask(self, question: str, history: list[dict[str, str]]) -> ResearchAnswer:
        """Answer one question using caller-owned history; retain no chat internally."""
        question = question.strip()
        if not question or len(question) > self.settings.max_question_chars:
            raise ResearchError(f"Enter a question of 1-{self.settings.max_question_chars} characters.")
        context = "\n".join(f"{item['role']}: {item['content']}" for item in history)
        # Retain the newest context within a character budget (not a token budget).
        try:
            result = self.executor.invoke({"input": question,
                "chat_history": context[-self.settings.max_history_chars:]})
        except Exception as exc:
            # Provider exceptions may include credentials or request contents.
            # Log only the exception class and show a safe message without a traceback.
            logger.warning("Agent request failed: %s", type(exc).__name__)
            raise ResearchError("Request failed. Check API access, model, quota, and connection; then retry.") from None
        output = str(result.get("output", "")).strip()
        # An executor stopping message is not a completed research answer.
        if not output or output.startswith("Agent stopped"):
            raise ResearchError("The research limit was reached. Try a narrower question.")
        # Remove duplicate tool names while preserving their first-use order.
        tools = tuple(dict.fromkeys(action.tool for action, _ in result.get("intermediate_steps", [])))
        return ResearchAnswer(output, tools)
