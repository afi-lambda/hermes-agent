"""
Compact, structured role prompts for the orchestration layer.

These are deliberately small: the architect does NOT write a long essay, the
worker is handed the original request + a compact brief, and the reviewer does
independent review without being asked to expose hidden chain-of-thought.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional


def _render_acceptance_criteria(criteria: Iterable[str]) -> str:
    items = list(criteria or [])
    if not items:
        return ""
    return "\n".join(f"  - {c}" for c in items)


def architect_prompt(original_request: str) -> str:
    """Prompt the architect to produce a compact structured implementation brief.

    The original user request is preserved as authoritative context.
    """
    return (
        "You are the architect for an implementation task.\n"
        "Produce a COMPACT, structured implementation brief only. Do not write a "
        "long essay, do not dump reasoning or internal chain-of-thought.\n\n"
        "Use this exact structure (each section one to a few lines):\n"
        "objective:\n"
        "constraints:\n"
        "files_components_likely_affected:\n"
        "implementation_approach:\n"
        "acceptance_criteria:\n"
        "risks:\n"
        "tests_required:\n\n"
        "ORIGINAL REQUEST (authoritative):\n"
        f"{original_request}\n"
    )


def worker_prompt(
    original_request: str,
    architect_brief: Optional[str] = None,
    acceptance_criteria: Iterable[str] = (),
    revision_context: Optional[str] = None,
) -> str:
    """Prompt the worker to implement the task.

    Receives the original request, an optional architect brief, explicit
    acceptance criteria, and optional reviewer feedback for a revision.
    """
    lines: List[str] = [
        "You are the implementing worker. Complete the task using your normal "
        "tools and repository access.",
        "",
        "ORIGINAL REQUEST (authoritative):",
        original_request,
        "",
    ]
    if architect_brief:
        lines += [
            "ARCHITECT BRIEF:",
            architect_brief,
            "",
        ]
    criteria = _render_acceptance_criteria(acceptance_criteria)
    if criteria:
        lines += [
            "ACCEPTANCE CRITERIA:",
            criteria,
            "",
        ]
    if revision_context:
        lines += [
            "REVISION FEEDBACK FROM REVIEWER (address these):",
            revision_context,
            "",
        ]
    lines += [
        "When finished, report in your summary: changed files, tests run and "
        "their results, and any remaining uncertainty.",
    ]
    return "\n".join(lines)


def reviewer_prompt(
    original_request: str,
    acceptance_criteria: Iterable[str],
    worker_summary: str,
    worker_model: Optional[str],
    tool_trace: Optional[List[Dict[str, Any]]],
    metrics: Optional[Dict[str, Any]] = None,
) -> str:
    """Prompt the reviewer to perform an independent review.

    The reviewer returns a STRICT JSON object (no prose, no hidden
    chain-of-thought). Focus: observable correctness, requirements, tests,
    regressions, safety, maintainability.
    """
    criteria = _render_acceptance_criteria(acceptance_criteria)
    tool_summary = _summarize_tool_trace(tool_trace)
    metrics_line = ""
    if metrics:
        metrics_line = (
            "\nOBSERVED EXECUTION METRICS:\n"
            f"{json.dumps(metrics, default=str, indent=2)}\n"
        )
    return (
        "You are an independent reviewer. Review the following task WITHOUT "
        "continuing the worker's chain of thought.\n\n"
        "ORIGINAL REQUEST:\n"
        f"{original_request}\n\n"
        f"ACCEPTANCE CRITERIA:\n{criteria or '  (none provided)'}\n\n"
        f"WORKER SUMMARY:\n{worker_summary}\n\n"
        f"WORKER MODEL: {worker_model or 'unknown'}\n"
        f"TOOL CALL TRACE:\n{tool_summary}\n"
        f"{metrics_line}"
        "\nInspect the repository as needed to verify correctness, "
        "requirements coverage, tests, regressions, safety, and maintainability. "
        "Inspect the changed files yourself rather than trusting the summary.\n\n"
        "Respond with ONLY a JSON object in this exact shape (no markdown fence, "
        "no extra text):\n"
        '{"verdict": "accept" | "revise" | "escalate",'
        ' "issues": ["..."],'
        ' "required_changes": ["..."],'
        ' "confidence": 0.0}\n'
        "- verdict accept: meets acceptance criteria, no blocking issues.\n"
        "- verdict revise: fixable issues; list them in required_changes.\n"
        "- verdict escalate: task grew / architecture unclear / repeated "
        "failures; escalate to the architect model.\n"
        "- confidence: your confidence in this verdict, 0.0 to 1.0.\n"
    )


def _summarize_tool_trace(tool_trace: Optional[List[Dict[str, Any]]]) -> str:
    """Render a compact, non-sensitive view of the worker's tool calls."""
    if not tool_trace:
        return "(none)"
    entries = []
    for item in tool_trace:
        if not isinstance(item, dict):
            continue
        tool = item.get("tool", "unknown")
        status = item.get("status", "")
        entries.append(f"  {tool}" + (f" [{status}]" if status else ""))
    # Keep it bounded.
    return "\n".join(entries[:40])
