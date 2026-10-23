from __future__ import annotations
from langchain_core.messages import AIMessage


def extract_tokens(message: AIMessage) -> int:
    """Extract total token count from an AIMessage's usage_metadata, or 0."""
    meta = getattr(message, "usage_metadata", None)
    if meta is None:
        return 0
    return meta.get("total_tokens", 0)
