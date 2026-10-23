"""
TruncationRecoveryMiddleware — detects when the model's last completion was cut off by
the per-turn token-budget limit (qwen38_max_tokens / primary_max_tokens), not a natural
stopping point, and re-triggers the agent with a targeted recovery message.

Real incident in the source research project: multiple episodes hit at least one
AIMessage with content="" and output_tokens exactly at the configured per-turn cap -- the
model's internal reasoning consumed the entire turn's output budget before ever
producing visible content or a tool call. Usually just a wasted turn, but one episode hit
it right after risk-model training, then again on both of ResultsGuardMiddleware's
recovery attempts, so risk_model.pkl was never written -- a total, unrecoverable
eval_status="failed" episode. Raising the per-turn token cap reduces how OFTEN this
fires, but doesn't structurally prevent it (any fixed budget still has some tail
probability of being hit), so this is the complementary backstop: catch it explicitly and
recover, the same role ResultsGuardMiddleware plays for missing/broken artifacts, not a
replacement for a large enough budget.

Detection: langchain_openai.ChatOpenAI does NOT extract/preserve vLLM's non-standard
`reasoning_content` field (stated in that library's own module docstring -- it targets
the official OpenAI API surface only), so a completion cut off mid-<think> block arrives
here as an AIMessage with content="" and no usable trace of what was being reasoned
through. What IS preserved is `finish_reason` in the message's response_metadata --
"length" when the server stopped generation purely because the token cap was hit, vs
"stop"/"tool_calls" for a natural or tool-call-terminated end. That's a hard, direct
signal from the server about *why* generation stopped, not something inferred from
content length or presence -- so it correctly catches BOTH (a) content fully empty
(reasoning ate the whole budget) and (b) content non-empty but itself truncated
mid-generation (reasoning ate most of the budget, a few tokens of real content escaped
before the cap hit) -- content length/presence never enters the detection condition.

Deliberately does NOT gate on "no tool_calls" the way an earlier draft of this design
did. A completion can be cut off by the token cap AFTER emitting a syntactically valid
tool call whose arguments were themselves truncated mid-generation (e.g. a write_file
call with file content cut off partway through) -- that parses as a normal-looking tool
call, so a "skip if tool_calls present" filter would silently miss it, and arguably this
is the more dangerous case: a truncated tool call can corrupt a saved file with no
visible error at all, unlike an empty response which is at least obviously inert. So the
sole trigger condition is finish_reason == "length"; the recovery message just branches
on whether a tool call is present, asking the agent to verify what it touched rather than
assuming it completed correctly.

Ordering: must sit BEFORE ResultsGuardMiddleware in the middleware list. Each
middleware's after_agent becomes its own chained node in the underlying graph
(langchain.agents.factory), and returning jump_to="model" short-circuits that chain for
this pass -- so whichever of these two fires first is the one that acts. Truncation is
the more specific diagnosis (this exact turn was cut short) vs ResultsGuard's broader
"are the deliverables present/valid" check, and fixing the more specific, more likely
root cause first gives the agent a cleaner recovery than a generic "you haven't saved X"
nudge would when the actual problem was a token cutoff.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, HumanMessage

log = logging.getLogger(__name__)

_NO_ACTION_MESSAGE = (
    "IMPORTANT: your last response was cut off by the per-turn token limit before "
    "producing any visible output (no answer text, no tool call) -- this is a hard "
    "cutoff on how long a single turn's generation can run, not a sign you did "
    "something wrong, and not something a longer explanation will fix. Whatever you "
    "were reasoning through, don't try to redo all of it here -- respond concisely "
    "this turn: skip lengthy internal deliberation and go straight to one concrete "
    "action (e.g. write a file, run a check, or save your current best artifacts)."
)

_TOOL_CALL_MESSAGE = (
    "IMPORTANT: your last response hit the per-turn token limit while it was still "
    "being generated. It did produce a tool call, but a response cut off "
    "mid-generation can leave that tool call's arguments truncated (e.g. a file write "
    "with content cut off partway through) even though the call itself looks "
    "syntactically valid and may have already run. Before continuing: verify whatever "
    "that last tool call touched actually completed correctly (re-read the file it "
    "wrote, re-run the check it performed) rather than assuming it worked."
)


class TruncationRecoveryMiddleware(AgentMiddleware):
    """Re-triggers the agent with a targeted recovery message if the last completion
    was cut off by the per-turn token-budget limit (finish_reason == "length") rather
    than a natural stop or tool-call completion."""

    def __init__(self, max_retries: int = 2) -> None:
        self.max_retries = max_retries
        self._attempts = 0

    @staticmethod
    def _last_ai_message(state: Any) -> AIMessage | None:
        messages = state.get("messages", []) if state is not None else []
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                return msg
        return None

    @hook_config(can_jump_to=["model"])
    def after_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        last = self._last_ai_message(state)
        if last is None:
            return None

        finish_reason = (last.response_metadata or {}).get("finish_reason")
        if finish_reason != "length":
            return None

        if self._attempts >= self.max_retries:
            log.warning(
                "[TruncationRecovery] finish_reason=length after %d retries — giving up",
                self._attempts,
            )
            return None

        self._attempts += 1
        has_tool_calls = bool(getattr(last, "tool_calls", None))
        content = _TOOL_CALL_MESSAGE if has_tool_calls else _NO_ACTION_MESSAGE
        log.warning(
            "[TruncationRecovery] finish_reason=length (tool_calls=%s) — injecting "
            "recovery prompt (attempt %d/%d)",
            has_tool_calls, self._attempts, self.max_retries,
        )
        return {
            "messages": [HumanMessage(content=content)],
            "jump_to": "model",
        }

    @hook_config(can_jump_to=["model"])
    async def aafter_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self.after_agent(state, runtime)
