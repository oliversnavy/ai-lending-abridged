from __future__ import annotations
import os
import pathlib

import yaml

from ..models.episode import RunConfig

_CONFIG: dict | None = None


def _cfg() -> dict:
    global _CONFIG
    if _CONFIG is None:
        path = pathlib.Path("configs/base.yaml")
        with open(path) as f:
            _CONFIG = yaml.safe_load(f)
    return _CONFIG


def get_primary_client(
    run_config: RunConfig,
    max_tokens_override: int | None = None,
    reasoning_effort: str | None = None,
):
    """Returns a LangChain chat model for `run_config.backend`.

    Both backends are hosted APIs -- no local model server to health-check or wait for
    (the source research project's harness runs against a self-hosted vLLM server and
    has a `wait_for_model_server` liveness probe for that; this repo intentionally
    doesn't ship that path -- see ARCHITECTURE.md).
    """
    m = _cfg()["models"]

    if run_config.backend == "baseten":
        # Baseten's OpenAI-compatible endpoint (GLM-5.3, open-weight) -- a documented
        # second backend, useful when a reviewer wants to see this harness run against
        # a self-hostable/open-weight model rather than a proprietary API.
        from langchain_openai import ChatOpenAI

        api_key = os.environ.get("BASETEN_API_KEY")
        if not api_key:
            raise RuntimeError(
                "BASETEN_API_KEY is not set -- required for backend='baseten'. "
                "Add it to .env and `source .env` before running."
            )
        extra_body = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
        return ChatOpenAI(
            base_url=m["baseten_url"],
            api_key=api_key,
            model=m["baseten_model"],
            max_tokens=max_tokens_override or m["baseten_max_tokens"],
            temperature=0.6,
            extra_body=extra_body,
        )

    # Default: Anthropic.
    from langchain_anthropic import ChatAnthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set -- required for backend='anthropic' "
            "(the default). Add it to .env and `source .env` before running."
        )
    # reasoning_effort is a native ChatAnthropic field (low/medium/high/xhigh/max) --
    # the exact same vocabulary this harness's RunConfig.reasoning_effort/
    # critic_reasoning_effort already use, confirmed live against the real API (Sonnet 5
    # requires thinking.type="adaptive" + output_config.effort for budget control, not
    # the older thinking.type="enabled" + budget_tokens scheme; passing reasoning_effort
    # directly lets langchain-anthropic handle that wiring itself).
    return ChatAnthropic(
        model=m["anthropic_model"],
        max_tokens=max_tokens_override or m["anthropic_max_tokens"],
        temperature=1,
        reasoning_effort=reasoning_effort or run_config.reasoning_effort,
    )


def get_post_mortem_max_tokens() -> int:
    return _cfg()["models"]["post_mortem_max_tokens"]


def get_advisor_client(run_config: RunConfig):
    return get_primary_client(run_config, reasoning_effort="medium")
