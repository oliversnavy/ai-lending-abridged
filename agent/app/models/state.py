from __future__ import annotations
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from .episode import EpisodeRecord, RunConfig


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    episode_id: int
    run_config: RunConfig
    episode_index: list[EpisodeRecord]
    skill_dir: str          # absolute path as str (JSON-serialisable)
    tokens_used: int
