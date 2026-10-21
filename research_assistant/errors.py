"""Application exceptions whose messages may be rendered directly in the UI."""


class ResearchError(Exception):
    """An error whose message is safe to show to the user."""


class ConfigurationError(ResearchError):
    """Missing or invalid configuration that must be fixed before startup."""


class DocumentError(ResearchError):
    """Unsupported, oversized, unreadable, or text-free document input."""
