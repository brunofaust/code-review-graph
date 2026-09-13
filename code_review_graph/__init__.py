"""Code Review Graph - MCP server for persistent incremental code knowledge graphs."""

from typing import Final

from .context_savings import (
    attach_context_savings,
    estimate_context_savings,
    estimate_file_tokens,
    estimate_tokens,
    format_context_savings,
)

__version__ = "2.3.8+bd.2"
BUSYDONE_FORK_MARKER: Final[str] = "code-review-graph-busydone-core"
BUSYDONE_FORK_VERSION: Final[str] = __version__
BUSYDONE_FORK: Final[tuple[str, str]] = (
    BUSYDONE_FORK_MARKER,
    BUSYDONE_FORK_VERSION,
)

__all__ = [
    "BUSYDONE_FORK",
    "BUSYDONE_FORK_MARKER",
    "BUSYDONE_FORK_VERSION",
    "__version__",
    "attach_context_savings",
    "estimate_context_savings",
    "estimate_file_tokens",
    "estimate_tokens",
    "format_context_savings",
]
