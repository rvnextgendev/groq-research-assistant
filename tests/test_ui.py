from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from research_assistant.agent import ResearchAnswer
from research_assistant.config import Settings


def test_missing_config_shows_help():
    from research_assistant.errors import ConfigurationError
    with patch("research_assistant.ui.Settings.load", side_effect=ConfigurationError("Configure API keys")):
        app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=15).run()
    assert not app.exception
    assert "Configure" in app.error[0].value


def test_upload_prompt_and_disabled_chat():
    with patch("research_assistant.ui.Settings.load", return_value=Settings("test", "test")):
        app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=15).run()
    assert not app.exception
    assert app.chat_input[0].disabled


def test_chat_and_download():
    class Service:
        def ask(self, question, history):
            return ResearchAnswer("Revenue grew 34% [PDF page 1]", ("pdf_knowledge_base",))
    with patch("research_assistant.ui.Settings.load", return_value=Settings("test", "test")):
        app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=15)
        app.session_state["service"] = Service()
        app.session_state["document_stats"] = "1 page"
        app.run()
        app.chat_input[0].set_value("What is revenue?").run()
        assert not app.exception
        assert len(app.chat_message) == 2
        assert "34%" in app.session_state["messages"][1]["content"]
        assert len(app.get("download_button")) == 1
        app.sidebar.button[1].click().run()
        assert app.session_state["messages"] == []


def test_switching_backend_clears_old_service():
    with patch("research_assistant.ui.Settings.load", return_value=Settings("test", "test")):
        app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=15)
        app.session_state["service"] = object()
        app.session_state["document_stats"] = "1 page"
        app.session_state["messages"] = [{"role": "user", "content": "old question"}]
        app.run()
        app.sidebar.selectbox[0].select("LangGraph").run()
        assert not app.exception
        assert app.chat_input[0].disabled
        assert app.session_state["messages"] == []
