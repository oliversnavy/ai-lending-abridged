from __future__ import annotations
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from ..prompts.advisor import ADVISOR_SYSTEM_PROMPT


def build_advisor_consult_tool(advisor_llm):
    """Return an advisor_consult tool bound to the given advisor LLM."""

    @tool
    def advisor_consult(context: str, question: str) -> str:
        """
        Consult the senior advisor on a strategic question.

        Consult at the START (plan review), at key DECISION POINTS (model choice,
        feature selection, how to handle censoring), when STUCK (poor c-stat, unclear
        cause), and at the END (results review). Aim for 2–4 consultations per episode.
        Keep queries specific.

        Args:
            context:  Brief summary of what you\'ve done so far and any relevant results.
            question: The specific question or decision you need guidance on.

        Returns:
            Advisor\'s recommendation.
        """
        messages = [
            SystemMessage(content=ADVISOR_SYSTEM_PROMPT),
            HumanMessage(content=f"## Context\n{context}\n\n## Question\n{question}"),
        ]
        response = advisor_llm.invoke(messages)
        return response.content

    return advisor_consult
