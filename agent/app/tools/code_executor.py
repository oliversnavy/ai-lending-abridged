from __future__ import annotations
import pathlib
import subprocess
import sys
import textwrap

from langchain_core.tools import tool


TIMEOUT_SECONDS = 120
PROJECT_ROOT = pathlib.Path("/home/oliversnavy/repos/ai-lending-abridged")


@tool
def code_executor(code: str, filename: str = "script.py") -> str:
    """
    Execute Python code in the agent\'s working directory sandbox.

    Args:
        code:     Python source code to run.
        filename: Filename to save the code as (default: script.py).

    Returns:
        Combined stdout + stderr output, plus exit code.
    """
    # skill_dir is injected at runtime — see tools/__init__.py get_tools()
    raise NotImplementedError("Use build_code_executor(skill_dir) to get a bound tool.")


def build_code_executor(skill_dir: pathlib.Path):
    """Return a code_executor tool bound to the given skill directory."""

    @tool
    def _execute(code: str, filename: str = "script.py") -> str:
        """
        Execute Python code in the sandbox working directory.

        Args:
            code:     Python source code to run.
            filename: Filename to save the script as (default: script.py).

        Returns:
            stdout + stderr output and exit code summary.
        """
        skill_dir.mkdir(parents=True, exist_ok=True)
        script_path = skill_dir / filename
        script_path.write_text(code)

        env = {
            # skill_dir FIRST, not PROJECT_ROOT -- a real incident in the source research
            # project: with PROJECT_ROOT first, a stray root-level model_def.py (old WIP
            # scratch content) shadowed the episode's own
            # skill_dir/model_def.py on `import model_def`, silently loading the wrong
            # class. Putting skill_dir first closes the general case,
            # matching the same fix already applied to
            # evaluate_episode.py's scoring subprocess (see that module's own PYTHONPATH
            # comment) -- an episode's own modules should always win over anything at
            # the project root, regardless of what happens to exist there later.
            "PYTHONPATH": f"{skill_dir}:{PROJECT_ROOT}",
            "LENDING_DATA_DIR": str(PROJECT_ROOT / "data" / "processed"),
            "PATH": "/usr/bin:/bin:/usr/local/bin",
        }

        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(skill_dir),
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
                env=env,
            )
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            exit_code = result.returncode

            parts = []
            if stdout:
                parts.append(f"STDOUT:\n{stdout}")
            if stderr:
                parts.append(f"STDERR:\n{stderr}")
            parts.append(f"EXIT CODE: {exit_code}")
            return "\n\n".join(parts) if parts else f"EXIT CODE: {exit_code} (no output)"

        except subprocess.TimeoutExpired:
            return f"ERROR: Script timed out after {TIMEOUT_SECONDS}s."
        except Exception as e:
            return f"ERROR: {e}"

    _execute.name = "code_executor"
    return _execute
