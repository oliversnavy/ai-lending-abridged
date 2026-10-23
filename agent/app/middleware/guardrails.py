"""
Guardrail middleware for deep-agent (use_deep_agent=True) runs.

Intercepts write_file, edit_file, execute, ls, read_file, glob, and grep tool calls
before they run and blocks anything that would:
  - Touch files outside the allowed directories (skill_dir, data/processed/)
  - Run destructive or network shell commands (rm, curl, git push, etc.)
  - Read (ls/read_file/glob/grep) another loop instance's or another run_label's
    saved episode files -- added 2026-09-11, see _build_instance_read_rules below.

Returns a structured error ToolMessage so the model can self-correct without
the harness crashing.
"""
from __future__ import annotations

import logging
import pathlib
import re
from typing import Any, Callable, Awaitable

from deepagents import FilesystemPermission
from deepagents.middleware.filesystem import _check_fs_permission
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

log = logging.getLogger(__name__)

PROJECT_ROOT = pathlib.Path("/home/oliversnavy/repos/ai-lending-abridged")
RESULTS_DIR = PROJECT_ROOT / "results"

# The results/ subtrees that ever contain per-episode or per-run_label content --
# anything under these outside the current instance's own allowed range is off-limits.
# Kept as resolved Path objects (not strings) so _check_search_root's ancestor check
# below can use pathlib.Path.is_relative_to, not fragile string prefixing.
#
# results/pilots added 2026-09-12: predicted as a real risk in known_limitations memory
# 44 days before this fix ("filesystem_read can access all of results/... code_executor
# is unrestricted Python -- an agent could pickle.load() a prior model if it found the
# path"), deferred at the time as low-probability. It's covered automatically on the
# execute()/SandboxedLocalShellBackend side (that fix hides all of results/ by default,
# only punching back through skills/traces), but was missing here on the native-tool
# side until this same pass caught it -- verified directly (_check_fs_permission
# returned "allow" for a results/pilots path before this line was added).
_RESTRICTED_READ_ROOTS: tuple[pathlib.Path, ...] = (
    RESULTS_DIR / "skills",
    RESULTS_DIR / "traces",
    RESULTS_DIR / "episode_indexes",
    RESULTS_DIR / "loop_state",
    RESULTS_DIR / "pilots",
)


def _build_instance_read_rules(
    run_label: str, episode_id_range: tuple[int, int],
) -> list[FilesystemPermission]:
    """Build allow/deny rules scoping ls/read_file/glob/grep to episodes within
    episode_id_range (inclusive) -- denying every other episode this run_label has ever
    run, and every other run_label entirely.

    Real incident in the source research project: nothing previously restricted these
    four tools' paths at all (this middleware only ever checked write_file/edit_file/
    execute, never reads) -- verified two real episodes used an unrestricted recursive
    glob to read a DIFFERENT loop instance's saved data_pipeline.py. What was actually
    read was inert boilerplate in those specific cases, but nothing structurally prevented
    reading a genuinely competitive risk_model.pkl the same way -- a
    real hole in the "N independent instances/episodes" assumption every run's
    reported statistic relies on.

    Reuses deepagents' own FilesystemPermission/_check_fs_permission -- NOT via
    create_deep_agent's permissions= (that raises NotImplementedError on any
    execute-capable backend, which SandboxedLocalShellBackend is -- see
    FilesystemMiddleware.__init__). _check_fs_permission itself is a plain function with
    no backend coupling, so calling it directly from here, inside our own
    already-established wrap_tool_call guardrail, sidesteps that restriction entirely.

    Rules are evaluated in order, first match wins (same semantics _check_fs_permission
    already implements) -- the two "allow" rules for this episode's own instance must
    come before the broader "deny" rules covering the rest of results/skills and
    results/traces.
    """
    lo, hi = episode_id_range
    own_skill_patterns = [
        str(RESULTS_DIR / "skills" / f"run_{run_label}" / f"episode_{i:04d}") + "{,/**}"
        for i in range(lo, hi + 1)
    ]
    own_trace_patterns = [
        str(RESULTS_DIR / "traces" / f"run_{run_label}" / f"episode_{i:04d}") + "*"
        for i in range(lo, hi + 1)
    ]
    return [
        FilesystemPermission(operations=["read"], paths=own_skill_patterns, mode="allow"),
        FilesystemPermission(operations=["read"], paths=own_trace_patterns, mode="allow"),
        FilesystemPermission(operations=["read"], paths=[str(RESULTS_DIR / "skills") + "/**"], mode="deny"),
        FilesystemPermission(operations=["read"], paths=[str(RESULTS_DIR / "traces") + "/**"], mode="deny"),
        # No legitimate need to read these directly -- everything they contain is
        # either this instance's own data (already available via the injected
        # ALL_PRIOR context) or another instance/run_label's, which is exactly what
        # this function exists to keep out of reach.
        FilesystemPermission(operations=["read"], paths=[str(RESULTS_DIR / "episode_indexes") + "/**"], mode="deny"),
        FilesystemPermission(operations=["read"], paths=[str(RESULTS_DIR / "loop_state") + "/**"], mode="deny"),
        # Pre-dates the current ladder entirely (pilot runs, never part of any live
        # instance) -- no "own instance" carve-out applicable, unconditional deny.
        FilesystemPermission(operations=["read"], paths=[str(RESULTS_DIR / "pilots") + "/**"], mode="deny"),
    ]

# Shell patterns that are never allowed, regardless of path.
# Note: bare `rm` (single file) within skill_dir is handled separately in
# _check_command() and is allowed. Only recursive/force deletes are caught here.
_BLOCKED_SHELL_PATTERNS: list[tuple[str, str]] = [
    (r"\brm\s+-[rRfFiI]*[rR][rRfFiI]*\b", "rm -r/-rf is not allowed — only single-file rm within your working directory is permitted"),
    (r"\brmdir\b", "rmdir is not allowed"),
    (r"\bcurl\b", "network access via curl is not allowed"),
    (r"\bwget\b", "network access via wget is not allowed"),
    (r"\bpip\b.*install", "pip install is not allowed during an episode"),
    (r"\bsudo\b", "sudo is not allowed"),
    (r"\bgit\s+(push|reset|checkout|rebase|merge|branch\s+-[Dd])", "destructive git operations are not allowed"),
    (r"\bchmod\b", "chmod is not allowed"),
    (r"\bchown\b", "chown is not allowed"),
    (r"\bkill\b|\bpkill\b|\bkillall\b", "killing processes is not allowed"),
    (r"\bcrontab\b", "crontab modification is not allowed"),
    (r"\bdd\b\s+if=", "dd is not allowed"),
    (r">\s*/dev/sd[a-z]", "writing to raw block devices is not allowed"),
]

# File-write tool names in deepagents
_WRITE_TOOL_NAMES = {"write_file", "edit_file"}

# Shell execution tool names
_EXEC_TOOL_NAMES = {"execute"}

# Read tool names in deepagents with a single, explicit target path (ls's "path" is a
# directory to list, read_file's "file_path" is a file to read -- both name the exact
# target, unlike glob/grep's "path" which is a SEARCH ROOT, see _GLOB_GREP_TOOL_NAMES).
_READ_TOOL_NAMES = {"ls": "path", "read_file": "file_path"}

# Bulk-search tools whose "path" argument is a recursive search ROOT, not a single
# target -- these need a broader check than _READ_TOOL_NAMES' exact-path match, since a
# root like PROJECT_ROOT can recurse into forbidden territory even though the root
# itself was never explicitly denied.
_GLOB_GREP_TOOL_NAMES = {"glob", "grep"}


def _violation_message(tool_name: str, reason: str) -> ToolMessage:
    return ToolMessage(
        content=(
            f"GUARDRAIL BLOCKED [{tool_name}]: {reason}\n"
            "Correction guidance:\n"
            "- To modify an existing file: use edit_file, not rm + write_file\n"
            "- To write new files: use write_file with a path inside your working directory\n"
            "- To run shell commands: only python3, standard data-science tools, and read operations are allowed"
        ),
        tool_call_id="__guardrail__",
        status="error",
    )


class HarnessGuardrailsMiddleware(AgentMiddleware):
    """
    Blocks file writes outside allowed directories, dangerous shell commands, and reads
    of another loop instance's or another run_label's saved episode files.

    Args:
        skill_dir: Episode working directory — the only place the agent may write.
        data_dir:  Read-only data directory — listed so ls() works, but write blocked.
        run_label: This episode's run_label key (e.g. "ltm_full") -- used to build the
            read-restriction rules below, scoped to this specific run_label's own
            results/skills and results/traces subtrees.
        instance_episode_id_range: (lo, hi) inclusive -- which of this run_label's
            episode ids remain readable beyond this episode's own. Always resolved by
            the caller (agent/app/agent.py::run_episode defaults it to
            (episode_id, episode_id) when not otherwise specified) -- never None here.
    """

    def __init__(
        self,
        skill_dir: pathlib.Path,
        data_dir: pathlib.Path,
        run_label: str,
        instance_episode_id_range: tuple[int, int],
    ) -> None:
        self._skill_dir = skill_dir.resolve()
        self._allowed_write = str(self._skill_dir)
        self._data_dir = str(data_dir.resolve())
        self._instance_lo, self._instance_hi = instance_episode_id_range
        self._read_rules = _build_instance_read_rules(run_label, instance_episode_id_range)

    # ------------------------------------------------------------------
    # Synchronous path (used by graph.stream / graph.invoke)
    # ------------------------------------------------------------------

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> Any:
        violation = self._check(request)
        if violation:
            tool_name = request.tool_call.get("name", "unknown")
            log.warning("[guardrail] blocked %s: %s", tool_name, violation)
            msg = _violation_message(tool_name, violation)
            # patch the tool_call_id so LangGraph can match it
            msg.tool_call_id = request.tool_call.get("id", "__guardrail__")
            return msg
        return handler(request)

    # ------------------------------------------------------------------
    # Async path (used by astream / ainvoke)
    # ------------------------------------------------------------------

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[Any]],
    ) -> Any:
        violation = self._check(request)
        if violation:
            tool_name = request.tool_call.get("name", "unknown")
            log.warning("[guardrail] blocked %s: %s", tool_name, violation)
            msg = _violation_message(tool_name, violation)
            msg.tool_call_id = request.tool_call.get("id", "__guardrail__")
            return msg
        return await handler(request)

    # ------------------------------------------------------------------
    # Internal checks
    # ------------------------------------------------------------------

    def _check(self, request: ToolCallRequest) -> str | None:
        """Return a violation reason string, or None if the call is allowed."""
        name = request.tool_call.get("name", "")
        args = request.tool_call.get("args", {})

        if name in _WRITE_TOOL_NAMES:
            return self._check_write_path(args.get("file_path") or args.get("path", ""))

        if name in _EXEC_TOOL_NAMES:
            return self._check_command(args.get("command", ""))

        if name in _READ_TOOL_NAMES:
            return self._check_read_path(args.get(_READ_TOOL_NAMES[name], ""))

        if name in _GLOB_GREP_TOOL_NAMES:
            return self._check_search_root(args.get("path"))

        return None

    def _check_write_path(self, path: str) -> str | None:
        if not path:
            return None
        resolved = str(pathlib.Path(path).resolve())
        if not resolved.startswith(self._allowed_write):
            return (
                f"write to '{path}' is outside your working directory "
                f"({self._allowed_write}). Write files there instead."
            )
        return None

    def _instance_denied_message(self, path: str) -> str:
        return (
            f"read access to '{path}' is outside this loop instance. You may read your "
            f"own working directory, other episodes within THIS instance "
            f"(episode ids {self._instance_lo:04d}-{self._instance_hi:04d}), and general "
            f"project code/data -- not other instances or other treatments' saved episodes."
        )

    def _check_read_path(self, path: str) -> str | None:
        """ls/read_file: a single, explicit target path -- direct reuse of
        _check_fs_permission, same pattern as _check_write_path above."""
        if not path:
            return None
        resolved = str(pathlib.Path(path).resolve())
        if _check_fs_permission(self._read_rules, "read", resolved) == "deny":
            return self._instance_denied_message(path)
        return None

    def _check_search_root(self, path: str | None) -> str | None:
        """glob/grep: "path" is a recursive SEARCH ROOT, not a single target -- a root
        can be denied two different ways:
          1. The root itself resolves inside a specific forbidden episode/run_label dir
             (same check as _check_read_path, e.g. path=".../episode_0082" directly).
          2. The root is an ANCESTOR of (or equal to) one of the four restricted
             results/ subtrees (e.g. path=PROJECT_ROOT, or bare results/skills) -- even
             though the root itself was never explicitly denied, a recursive search
             from there could still reach forbidden content beneath it. This is exactly
             the real incident that motivated this whole fix: episode 0088's exploit was
             glob(pattern='**/data_pipeline*.py', path=PROJECT_ROOT) -- PROJECT_ROOT
             itself isn't "under results/skills" so case 1 alone would have missed it.
        No path (None) means "search from the backend's default root" -- that's
        skill_dir for this backend, always safe, no check needed.
        """
        if not path:
            return None
        resolved_path = pathlib.Path(path).resolve()
        resolved = str(resolved_path)
        if _check_fs_permission(self._read_rules, "read", resolved) == "deny":
            return self._instance_denied_message(path)
        for restricted_root in _RESTRICTED_READ_ROOTS:
            if restricted_root.is_relative_to(resolved_path):
                return (
                    f"searching from '{path}' is too broad -- it could recurse into "
                    f"other instances' or treatments' saved episodes. Narrow the search "
                    f"to your own working directory, a specific episode within this "
                    f"instance, or a non-results project directory "
                    f"(e.g. data_pipeline/, agent/)."
                )
        return None

    def _check_command(self, command: str) -> str | None:
        # Special case: bare `rm <path>` — allow within skill_dir, block outside.
        # This lets the agent delete its own working files (e.g., to replace a script)
        # without being able to touch harness code or system files.
        bare_rm = re.search(r'\brm\s+(?!-)([\S]+)', command)
        if bare_rm and not re.search(r'\brm\s+-', command):
            target = bare_rm.group(1)
            if target.startswith('/'):
                resolved = str(pathlib.Path(target).resolve())
            else:
                resolved = str((self._skill_dir / target).resolve())
            if not resolved.startswith(self._allowed_write):
                return (
                    f"rm '{target}' is outside your working directory "
                    f"({self._allowed_write}). You may only delete files you created."
                )
            return None  # rm within skill_dir is allowed

        for pattern, reason in _BLOCKED_SHELL_PATTERNS:
            if re.search(pattern, command):
                return reason
        return None
