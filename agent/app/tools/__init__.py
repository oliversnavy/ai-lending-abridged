from __future__ import annotations
import pathlib

from langchain_core.tools import BaseTool

from ..models.episode import RunConfig
from .code_executor import build_code_executor
from .filesystem import build_filesystem_tools


def get_t0_tools(run_config: RunConfig, skill_dir: pathlib.Path) -> list[BaseTool]:
    """Full tool set for the vanilla (use_deep_agent=False) create_agent path."""
    fs_read, fs_write = build_filesystem_tools(skill_dir)
    return [
        build_code_executor(skill_dir),
        fs_read,
        fs_write,
    ]


def get_supplementary_tools(run_config: RunConfig) -> list[BaseTool]:
    """
    Supplementary tools added on top of create_deep_agent built-ins.

    deepagents provides: ls, read_file, write_file, edit_file, glob, grep,
    execute, write_todos, task. We add only what it doesn't provide.
    """
    tools: list[BaseTool] = []

    if run_config.use_advisor:
        from ..clients.llm import get_advisor_client
        from .advisor_consult import build_advisor_consult_tool
        tools.append(build_advisor_consult_tool(get_advisor_client(run_config)))

    return tools
