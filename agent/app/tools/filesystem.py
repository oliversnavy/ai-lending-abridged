from __future__ import annotations
import pathlib

from langchain_core.tools import tool

PROJECT_ROOT = pathlib.Path("/home/oliversnavy/repos/ai-lending-abridged")


def build_filesystem_tools(skill_dir: pathlib.Path):
    """Return filesystem_read and filesystem_write tools bound to skill_dir.

    Read access is scoped to data/ (canonical, read-only project data) plus this
    episode's OWN skill_dir -- NOT the whole results/ tree. A run with no loop
    (loop_mode=NONE, no ALL_PRIOR injection) has no loop/instance concept at all, so
    every episode is meant to be fully independent; a prior version of this function
    let any such episode read any OTHER episode's saved files across every run_label
    and instance ever run -- the equivalent gap in the deep-agent path's
    deepagents-based read tools is closed in
    agent/app/middleware/guardrails.py::_build_instance_read_rules; same root problem,
    different tool implementation.
    """
    read_roots = [PROJECT_ROOT / "data", skill_dir.resolve()]

    @tool
    def filesystem_read(path: str) -> str:
        """
        Read a file from the data/ directory or your own working directory.

        Args:
            path: File path, relative to the project root or absolute.
        """
        p = pathlib.Path(path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        p = p.resolve()

        if not any(str(p).startswith(str(r)) for r in read_roots):
            return f"ERROR: Read access denied outside data/ and your own working directory. Got: {p}"

        if not p.exists():
            return f"ERROR: File not found: {p}"

        try:
            return p.read_text()
        except Exception as e:
            return f"ERROR reading {p}: {e}"

    @tool
    def filesystem_write(path: str, content: str) -> str:
        """
        Write a file inside the current episode\'s working directory.

        Args:
            path:    Filename or relative path (must stay within skill_dir).
            content: Text content to write.
        """
        p = pathlib.Path(path)
        if p.is_absolute():
            target = p
        else:
            target = skill_dir / p

        # Resolve and check confinement
        try:
            target.resolve().relative_to(skill_dir.resolve())
        except ValueError:
            return f"ERROR: Write access denied outside skill directory. Got: {target}"

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return f"OK: wrote {len(content)} bytes to {target}"

    return filesystem_read, filesystem_write
