class ResearchError(Exception):
    """An error whose message is safe to show to the user."""


class ConfigurationError(ResearchError):
    pass


class DocumentError(ResearchError):
    pass
