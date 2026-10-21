import pytest

from research_assistant.config import Settings


@pytest.fixture
def settings():
    return Settings("test-groq", "test-serper")
