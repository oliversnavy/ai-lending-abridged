"""
Pathological tool-error rate — a trace-quality diagnostic. Deliberately narrow: targets
runs of consecutive IDENTICAL tool calls that each fail, the signature of a genuinely
stuck agent (observed in the source research project in an episode that required manual
intervention) — NOT ordinary try-fail-fix debugging cycles (varying args/approach
between attempts), which is healthy, expected agent behavior and must not be penalized.

Reads the same trace JSONL files agent/app/agent.py's _write_trace already produces
(results/traces/run_{run_label}/episode_XXXX.jsonl), so this needs no new instrumentation.

Error detection is a string-match heuristic on the ToolMessage content following a
repeated call — informal by construction. Validate against a handful of real trace
files before trusting this as any kind of gate on a real decision.
"""
from __future__ import annotations

import json
import pathlib
import re

# Validated against real traces in the source research project: a naive substring match
# on "error"/"failed"/"exception" flagged 16-40% of ALL tool results as "errors" —
# almost entirely a single harmless environment quirk (matplotlib can't write its
# sandboxed home config dir, prints "mkdir -p failed ... Permission denied" and "there
# was an issue" to stderr, then falls back to /tmp and succeeds anyway) rather than
# genuine failures. Real failures (Python tracebacks, non-zero exit codes) are a small
# minority of that noise. Narrowed to a small set of precise, low-false-positive markers
# instead.
_STRONG_ERROR_PATTERNS = (
    re.compile(r"Traceback \(most recent call last\)"),
    re.compile(r"\[Command failed with exit code"),
    re.compile(r"(?m)^\w*(?:Error|Exception):"),  # e.g. "ValueError:", "AttributeError:" — a traceback's final line
)
def _args_signature(tool_call: dict) -> str:
    return json.dumps(tool_call.get("args", {}), sort_keys=True, default=str)


def _looks_like_error(content: object) -> bool:
    text = str(content)
    return any(pattern.search(text) for pattern in _STRONG_ERROR_PATTERNS)


def _tool_invocation_sequence(trace_path: pathlib.Path) -> list[tuple[str, bool]]:
    """Returns [(tool_name + "|" + args_signature, is_error), ...] in trace order, one
    entry per tool call, matched to its ToolMessage result by tool_call_id."""
    entries = []
    for line in trace_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    # Map tool_call_id -> ToolMessage content, for lookup when we hit the issuing AIMessage.
    results_by_call_id: dict[str, object] = {}
    for entry in entries:
        if entry.get("type") == "ToolMessage" and entry.get("tool_call_id"):
            results_by_call_id[entry["tool_call_id"]] = entry.get("content")

    sequence: list[tuple[str, bool]] = []
    for entry in entries:
        if entry.get("type") != "AIMessage":
            continue
        for tool_call in entry.get("tool_calls", []) or []:
            call_id = tool_call.get("id")
            signature = f"{tool_call.get('name')}|{_args_signature(tool_call)}"
            result_content = results_by_call_id.get(call_id)
            is_error = result_content is not None and _looks_like_error(result_content)
            sequence.append((signature, is_error))
    return sequence


def has_pathological_run(trace_path: pathlib.Path, min_run: int) -> bool:
    """True iff the trace contains >= min_run consecutive tool calls with an IDENTICAL
    (tool_name, args) signature, each of which resulted in an error-looking ToolMessage —
    the signature of a genuinely stuck agent, not varied try-fail-fix debugging."""
    if not trace_path.exists():
        return False
    sequence = _tool_invocation_sequence(trace_path)
    if not sequence:
        return False

    run_signature: str | None = None
    run_length = 0
    for signature, is_error in sequence:
        if is_error and signature == run_signature:
            run_length += 1
        elif is_error:
            run_signature = signature
            run_length = 1
        else:
            run_signature = None
            run_length = 0
        if run_length >= min_run:
            return True
    return False


def pathological_error_rate(trace_paths: list[pathlib.Path], min_run: int = 3) -> float:
    """Fraction of the given episodes' traces containing >=1 pathological run."""
    if not trace_paths:
        return 0.0
    flagged = sum(1 for p in trace_paths if has_pathological_run(p, min_run))
    return flagged / len(trace_paths)
