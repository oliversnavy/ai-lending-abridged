"""
OverflowRecoveryMiddleware — catches context overflow 400 errors and retries.

This is the last-resort safety net after SummarizationMiddleware (state-level
compression) and ContextEditingMiddleware (proactive token-based clearing) have
both failed to prevent an overflow. When the model API returns a 400 indicating
the context limit was exceeded, this middleware progressively clears the oldest
uncleared tool result and retries, looping until the request fits or retries
are exhausted.

It must be the LAST wrap_model_call middleware in the stack so it sits
innermost — closest to the actual API call — and catches the real exception
before other wrappers see it.
"""
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)

log = logging.getLogger(__name__)

_OVERFLOW_CLEARED = "[overflow-cleared]"
_DEFAULT_MAX_RETRIES = 10


def _is_context_overflow(e: Exception) -> bool:
    msg = str(e).lower()
    return (
        "context limit" in msg
        or ("input tokens" in msg and "output tokens" in msg)
        or "maximum context length" in msg
        or "maximum_context_length" in msg
    )


def _clear_oldest_tool_result(messages: list[AnyMessage]) -> tuple[list[AnyMessage], bool]:
    """Replace the oldest uncleared ToolMessage content with a placeholder.

    Returns (new_messages, did_clear). If no clearable ToolMessage exists,
    returns the original list and False.
    """
    for i, msg in enumerate(messages):
        if isinstance(msg, ToolMessage) and msg.content != _OVERFLOW_CLEARED:
            new_messages = list(messages)
            new_messages[i] = msg.model_copy(update={"content": _OVERFLOW_CLEARED})
            return new_messages, True
    return messages, False


class OverflowRecoveryMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    """Reactive context overflow recovery via progressive tool-result clearing.

    Must be placed LAST among wrap_model_call middleware in the stack so it
    is the innermost wrapper and catches the API exception directly.
    """

    def __init__(self, max_retries: int = _DEFAULT_MAX_RETRIES) -> None:
        super().__init__()
        self.max_retries = max_retries

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT] | AIMessage:
        messages = list(request.messages)

        for attempt in range(self.max_retries + 1):
            try:
                return handler(request.override(messages=messages))
            except Exception as e:
                if not _is_context_overflow(e):
                    raise
                messages, cleared = _clear_oldest_tool_result(messages)
                if not cleared:
                    log.error(
                        "[OverflowRecovery] Context overflow but no tool results left to clear — giving up"
                    )
                    raise
                log.warning(
                    "[OverflowRecovery] Context overflow on attempt %d/%d — cleared oldest tool result, retrying",
                    attempt + 1,
                    self.max_retries,
                )

        log.error(
            "[OverflowRecovery] Context overflow persisted after %d retries — giving up",
            self.max_retries,
        )
        raise RuntimeError(
            f"Context overflow persisted after {self.max_retries} retries with progressive tool clearing"
        )

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT] | AIMessage:
        messages = list(request.messages)

        for attempt in range(self.max_retries + 1):
            try:
                return await handler(request.override(messages=messages))
            except Exception as e:
                if not _is_context_overflow(e):
                    raise
                messages, cleared = _clear_oldest_tool_result(messages)
                if not cleared:
                    log.error(
                        "[OverflowRecovery] Context overflow but no tool results left to clear — giving up"
                    )
                    raise
                log.warning(
                    "[OverflowRecovery] Context overflow on attempt %d/%d — cleared oldest tool result, retrying",
                    attempt + 1,
                    self.max_retries,
                )

        log.error(
            "[OverflowRecovery] Context overflow persisted after %d retries — giving up",
            self.max_retries,
        )
        raise RuntimeError(
            f"Context overflow persisted after {self.max_retries} retries with progressive tool clearing"
        )
