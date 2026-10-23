"""
SandboxedLocalShellBackend — LocalShellBackend with execute() run inside an
ephemeral Docker container instead of directly on the host.

Background (real incident in the source research project): an episode hit an import
error on one of the harness's own shared modules and "fixed" it by running
`cat > /home/.../data_pipeline/some_module.py << EOF` via the execute() tool —
overwriting the real, shared project module with a short stub.
HarnessGuardrailsMiddleware only pattern-matches a blocklist of
dangerous commands (rm -rf, curl, sudo, ...); it never checked whether a
shell command's output redirection targeted a path outside skill_dir, so an
absolute-path `cat > ... << EOF` sailed straight through. That's a structural
gap in LocalShellBackend itself, not just a missing regex: the upstream
library documents that `virtual_mode` and path-based guardrails provide "NO
security with shell access enabled, since commands can access any path on
the system."

Only execute() is overridden here. write_file/read_file/edit_file/ls/glob/
grep are inherited from FilesystemBackend (direct host filesystem I/O via
Python, not shell) and are unchanged — those are already correctly gated by
HarnessGuardrailsMiddleware._check_write_path(), which restricts them to
skill_dir. The gap was specifically shell commands bypassing that check
entirely, so this only needed to sandbox execute().

Each execute() call runs in a fresh, ephemeral `docker run --rm` container:
  - project_root, /usr, /etc bind-mounted read-only — same read access the
    agent already had (this does not restrict what the agent can read, only
    what it can write)
  - results/ specifically is then HIDDEN (an empty --tmpfs overlay) and
    selectively punched back through: this episode's own skill_dir (read-
    write) plus every OTHER episode within this SAME loop instance's own
    skill_dir + trace files (read-only) -- see _build_scoped_results_mount_args.
    Added after finding (empirically, across real traces in the source research
    project) that execute() shell commands were routinely used
    to read/copy files from OTHER instances and OTHER runs entirely --
    every observed case only touched a generic pickling
    stub, never a genuinely competitive risk_model.pkl,
    but nothing structurally prevented the latter. This
    closes the SAME "N independent instances/episodes" gap
    HarnessGuardrailsMiddleware's read-path checks close for the native
    ls/read_file/glob/grep tools -- execute() needed a mount-level fix instead
    of a command-parsing one, since robustly recognizing "does this arbitrary
    shell command reference a forbidden path" via regex isn't reliable (too
    many ways to reference a path in shell), whereas a mount either exists or
    doesn't, independent of how the command tries to reach it.
  - skill_dir bind-mounted read-write on top of the read-only project_root
    mount — the only writable path inside the sandbox
  - --network none — structurally blocks curl/wget/exfiltration at the
    kernel level, regardless of the regex blocklist
  - PATH is set explicitly with {project_root}/.venv/bin first, so bare
    `python3` in agent-authored commands resolves to the project's venv
    (matching existing behavior when the harness itself runs under `uv run`)

Any write outside skill_dir now fails at the kernel/mount level ("Read-only
file system"), not via pattern matching — this closes the class of bug, not
just the one command that triggered it. Same now holds for reads outside
this instance's own episodes under results/ — verified directly (see the
class docstring below) that Docker resolves the nested tmpfs-hide + punched-
through bind mounts correctly regardless of mount-flag order: a more
specific (longer/nested) destination always wins over an ancestor mount.

Chosen over bubblewrap: bwrap's unprivileged sandboxing is blocked on this
host by Ubuntu's apparmor_restrict_unprivileged_userns policy (no bwrap
AppArmor profile installed); enabling it needs a system-wide sudo policy
change. Docker requires no privilege change (the harness user is already in
the `docker` group) at the cost of a per-command container-startup latency
(~1-2s), which is negligible against the existing 180s execute() timeout.

max_output_bytes default is lowered from LocalShellBackend's 100_000 to
20_000: a single execute() call returning a ~100KB traceback
(e.g. a pandas KeyError dumping its entire missing-keys array, as was
observed repeatedly in the source research project) can push a conversation
from comfortably under budget to over the model's context limit in one step.
ContextEditingMiddleware (see graph.py) reacts to *accumulated* conversation
size using an approximate token counter that undercounts dense/repetitive
text worse than the ~2x factor it's tuned for, so it isn't reliable against
a single oversized spike -- it can only clean up growth it already knows
about, not a message that's already too large to safely add. Capping single
tool outputs well below the trigger threshold prevents any one call from
single-handedly causing the overflow, independent of counter accuracy.
"""
from __future__ import annotations

import os
import pathlib
import re
import subprocess
import uuid

from deepagents.backends.local_shell import LocalShellBackend
from deepagents.backends.protocol import ExecuteResponse

_SANDBOX_IMAGE = "ubuntu:24.04"
_STARTUP_OVERHEAD_SECONDS = 15  # docker run cold-start budget, added on top of the command's own timeout

_EPISODE_ID_RE = re.compile(r"episode_(\d+)$")


class SandboxedLocalShellBackend(LocalShellBackend):
    """LocalShellBackend whose execute() runs inside a locked-down, ephemeral Docker container.

    Accepts the same constructor arguments as LocalShellBackend, plus three
    required keywords:
      project_root: the read-only mount covering everything the agent should
        still be able to read (data, source, venv) but never write outside of
        root_dir (skill_dir).
      run_label, instance_episode_id_range: which of this run_label's episodes
        (beyond this one's own root_dir) remain readable inside the sandbox --
        see _build_scoped_results_mount_args. Always resolved by the caller
        (agent/app/agent.py::run_episode defaults instance_episode_id_range to
        (episode_id, episode_id) when not otherwise specified), matching the
        same contract HarnessGuardrailsMiddleware uses for the native
        ls/read_file/glob/grep tools.
    """

    def __init__(
        self,
        *args,
        project_root: pathlib.Path,
        run_label: str,
        instance_episode_id_range: tuple[int, int],
        max_output_bytes: int = 20_000,
        **kwargs,
    ) -> None:
        super().__init__(*args, max_output_bytes=max_output_bytes, **kwargs)
        self._project_root = pathlib.Path(project_root).resolve()
        self._venv_bin = self._project_root / ".venv" / "bin"
        self._run_label = run_label
        self._instance_lo, self._instance_hi = instance_episode_id_range

    def _own_episode_id(self) -> int | None:
        m = _EPISODE_ID_RE.search(str(self.cwd))
        return int(m.group(1)) if m else None

    def _build_scoped_results_mount_args(self) -> list[str]:
        """Hide the whole results/ tree inside the sandbox (--tmpfs, ephemeral --
        writes to it inside the throwaway container never touch the real host
        filesystem), then punch back through ONLY this instance's own allowed
        episodes: every OTHER episode id in [instance_lo, instance_hi] (this
        episode's own root_dir is mounted read-write separately, unchanged),
        both its skill_dir (read-only) and any trace files that already exist
        for it (main, subagent, post-mortem, critic variants).

        Verified directly (real `docker run`, not assumed) that Docker resolves
        this correctly regardless of mount-flag order: a bind mount targeting a
        path NESTED inside an ancestor --tmpfs always wins over that ancestor,
        so the current episode's own skill_dir ends up writable and each
        allowed sibling ends up readable, while anything not explicitly
        punched through (a different instance's or run_label's episode,
        results/episode_indexes, results/loop_state, results/archive, etc.)
        is genuinely absent -- `ls` on a hidden path's parent doesn't even
        list it, not just deny reading its content.
        """
        results_dir = self._project_root / "results"
        args = ["--tmpfs", str(results_dir)]
        own_id = self._own_episode_id()
        skills_root = results_dir / "skills" / f"run_{self._run_label}"
        traces_root = results_dir / "traces" / f"run_{self._run_label}"
        for i in range(self._instance_lo, self._instance_hi + 1):
            if i == own_id:
                continue
            sibling_skill_dir = skills_root / f"episode_{i:04d}"
            if sibling_skill_dir.is_dir():
                args += ["-v", f"{sibling_skill_dir}:{sibling_skill_dir}:ro"]
            if traces_root.is_dir():
                for trace_path in sorted(traces_root.glob(f"episode_{i:04d}*")):
                    args += ["-v", f"{trace_path}:{trace_path}:ro"]
        return args

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        if not command or not isinstance(command, str):
            return ExecuteResponse(output="Error: Command must be a non-empty string.", exit_code=1, truncated=False)

        effective_timeout = timeout if timeout is not None else self._default_timeout
        if effective_timeout <= 0:
            msg = f"timeout must be positive, got {effective_timeout}"
            raise ValueError(msg)

        skill_dir = str(self.cwd)
        container_name = f"sandbox-{uuid.uuid4().hex[:12]}"
        container_path = (
            f"{self._venv_bin}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        )
        docker_cmd = [
            "docker", "run", "--rm",
            "--name", container_name,
            "--network", "none",
            "--memory", "16g",
            # Without --user, containers run as root by default, so anything the
            # agent writes under skill_dir (a host bind mount, not a virtualized
            # fs) comes back root-owned. Running as the host uid:gid keeps
            # episode artifacts owned by the harness user, matching every other
            # write path (write_file/edit_file) and every pre-sandbox episode.
            "--user", f"{os.getuid()}:{os.getgid()}",
            "-v", f"{self._project_root}:{self._project_root}:ro",
            "-v", "/usr:/usr:ro",
            "-v", "/etc:/etc:ro",
            *self._build_scoped_results_mount_args(),
            "-v", f"{skill_dir}:{skill_dir}:rw",
            "-w", skill_dir,
            "-e", f"PATH={container_path}",
            _SANDBOX_IMAGE,
            "/bin/sh", "-c", command,
        ]

        try:
            result = subprocess.run(  # noqa: S603
                docker_cmd,
                check=False,
                capture_output=True,
                stdin=subprocess.DEVNULL,
                text=True,
                timeout=effective_timeout + _STARTUP_OVERHEAD_SECONDS,
                env=self._env,
            )
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "kill", container_name], capture_output=True)  # noqa: S603, S607
            msg = (
                f"Error: Command timed out after {effective_timeout} seconds. "
                "For long-running commands, re-run using the timeout parameter."
            )
            return ExecuteResponse(output=msg, exit_code=124, truncated=False)
        except Exception as e:  # noqa: BLE001
            return ExecuteResponse(
                output=f"Error executing command ({type(e).__name__}): {e}",
                exit_code=1,
                truncated=False,
            )

        output_parts = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            stderr_lines = result.stderr.strip().split("\n")
            output_parts.extend(f"[stderr] {line}" for line in stderr_lines)
        output = "\n".join(output_parts) if output_parts else "<no output>"

        truncated = False
        if len(output) > self._max_output_bytes:
            output = output[: self._max_output_bytes]
            output += f"\n\n... Output truncated at {self._max_output_bytes} bytes."
            truncated = True

        if result.returncode != 0:
            output = f"{output.rstrip()}\n\nExit code: {result.returncode}"

        return ExecuteResponse(output=output, exit_code=result.returncode, truncated=truncated)


__all__ = ["SandboxedLocalShellBackend"]
