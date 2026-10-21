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
    text: str
    tools_used: tuple[str, ...]


class ResearchService:
    def __init__(self, executor: AgentExecutor, settings: Settings):
        self.executor = executor
        self.settings = settings

    @classmethod
    def create(cls, index: DocumentIndex, settings: Settings) -> "ResearchService":
        llm = ChatGroq(api_key=settings.groq_api_key, model=settings.model,
                       temperature=0, max_tokens=2048, timeout=45, max_retries=1)
        tools = build_tools(index, settings)
        agent = create_react_agent(llm, tools, RESEARCH_PROMPT)
        executor = AgentExecutor(agent=agent, tools=tools, verbose=False,
            max_iterations=8, max_execution_time=120, handle_parsing_errors=True,
            return_intermediate_steps=True)
        return cls(executor, settings)

    def ask(self, question: str, history: list[dict[str, str]]) -> ResearchAnswer:
        question = question.strip()
        if not question or len(question) > self.settings.max_question_chars:
            raise ResearchError(f"Enter a question of 1-{self.settings.max_question_chars} characters.")
        context = "\n".join(f"{item['role']}: {item['content']}" for item in history)
        try:
            result = self.executor.invoke({"input": question,
                "chat_history": context[-self.settings.max_history_chars:]})
        except Exception as exc:
            logger.warning("Agent request failed: %s", type(exc).__name__)
            raise ResearchError("Request failed. Check API access, model, quota, and connection; then retry.") from None
        output = str(result.get("output", "")).strip()
        if not output or output.startswith("Agent stopped"):
            raise ResearchError("The research limit was reached. Try a narrower question.")
        tools = tuple(dict.fromkeys(action.tool for action, _ in result.get("intermediate_steps", [])))
        return ResearchAnswer(output, tools)
