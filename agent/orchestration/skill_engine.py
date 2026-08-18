"""
Solution 2 engine: thin Python core + a skill file.

Design (compared to the all-Python Solution 1 in ``orchestrator.py``):
- The Python here only enforces the **deterministic skeleton** — routing, the
  bounded worker<->reviewer loop, escalation depth, and metrics collection.
- The **behavioral guidance and prompt templates** (how to plan, how to
  implement, how to review, what JSON the reviewer must return) live in a skill
  file (``multi-model-orchestration``), not in hard-coded Python strings. The
  skill is loaded via the Hermes skill registry so it is editable as prose and
  portable to other agent hosts.

Rationale: the original Sol-Advisor was a markdown file instructing the agent.
The all-Python approach gives deterministic control flow + reliable metrics but
bakes the guidance into code. This engine keeps the deterministic loop in code
while moving the flexible, prose-editable guidance into a skill — a middle path
for benchmarking the two designs head-to-head.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.orchestration.config import (
    resolve_role_model_config,
    build_role_reasoning_config,
)
from agent.orchestration.router import (
    ARCHITECT_PLAN,
    ARCHITECT_SOLO,
    WORKER_IMPLEMENT_REVIEW,
    WORKER_SOLO,
    RoutingDecision,
)

logger = logging.getLogger(__name__)

ROLE_WORKER = "worker"
ROLE_ARCHITECT = "architect"
ROLE_REVIEWER = "reviewer"

VERDICT_ACCEPT = "accept"
VERDICT_REVISE = "revise"
VERDICT_ESCALATE = "escalate"
_VALID_VERDICTS = {VERDICT_ACCEPT, VERDICT_REVISE, VERDICT_ESCALATE}

# Strategy names must mirror router constants.
_STRATEGY_RUNNERS = {}


def run_skill_orchestrated(
    parent_agent: Any,
    prompt: str,
    orch_cfg: Dict[str, Any],
    decision: Optional[RoutingDecision] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the skill-driven orchestration and return a run_conversation-shaped dict."""
    if metrics is None:
        metrics = {}

    if decision is None:
        from agent.orchestration.router import route

        decision = route(prompt, orch_cfg)

    start = time.monotonic()
    m: Dict[str, Any] = {
        "strategy": decision.strategy,
        "worker_reasoning_effort": decision.worker_reasoning_effort,
        "review_required": decision.review_required,
        "reason": decision.reason,
        "roles": {},
        "review_iterations": 0,
        "escalations": 0,
        "verdicts": [],
        "tool_calls": 0,
        "wall_clock_seconds": 0.0,
        "total_api_calls": 0,
        "total_tokens": {"input": 0, "output": 0},
        "engine": "skill",
    }

    logger.info(
        "[orchestrator/skill] strategy=%s worker_effort=%s review=%s",
        decision.strategy,
        decision.worker_reasoning_effort,
        decision.review_required,
    )

    try:
        guidance = _load_skill_guidance()
        result = _dispatch(
            parent_agent,
            prompt,
            orch_cfg,
            decision,
            guidance,
            m,
        )
    except Exception as exc:  # noqa: BLE001 - never break the turn
        logger.warning("[orchestrator/skill] fatal error: %s", exc)
        result = {
            "ok": False,
            "final_response": f"[orchestration failed] {exc}",
            "error": str(exc),
        }

    m["wall_clock_seconds"] = round(time.monotonic() - start, 3)
    metrics.update(m)

    return {
        "final_response": result.get("final_response") or "",
        "messages": [],
        "api_calls": m["total_api_calls"],
        "completed": bool(result.get("ok")),
        "failed": not bool(result.get("ok")),
        "orchestration": m,
    }


def _load_skill_guidance() -> Dict[str, str]:
    """Locate and read the orchestration skill's SKILL.md.

    Searches the standard skill roots in priority order: the current repo's
    ``skills/`` tree, then the user's ``~/.hermes/skills/`` tree (via
    ``get_hermes_home``). Returns ``{"markdown": <body>}`` when found, else an
    empty dict (the engine falls back to compact built-in prompt templates so it
    still works without the skill installed).
    """
    roots = []
    try:
        roots.append(Path(__file__).resolve().parent.parent.parent / "skills")
    except Exception:
        pass
    try:
        from hermes_constants import get_hermes_home

        roots.append(Path(get_hermes_home()) / "skills")
    except Exception:
        pass

    for root in roots:
        if not root.is_dir():
            continue
        for candidate in root.rglob("multi-model-orchestration/SKILL.md"):
            try:
                content = candidate.read_text(encoding="utf-8")
            except OSError:
                continue
            body = _strip_frontmatter(content)
            if body:
                return {"markdown": body}
    return {}


def _strip_frontmatter(content: str) -> str:
    """Remove the YAML frontmatter block, returning the markdown body."""
    if content.startswith("---"):
        # find the closing '---' fence after the opening one
        idx = content.find("\n---", 3)
        if idx != -1:
            body = content[idx + 4 :]
            return body.strip()
    return content.strip()


def _dispatch(
    parent_agent: Any,
    prompt: str,
    orch_cfg: Dict[str, Any],
    decision: RoutingDecision,
    guidance: Dict[str, str],
    m: Dict[str, Any],
) -> Dict[str, Any]:
    strategy = decision.strategy
    if strategy == ARCHITECT_SOLO:
        return _architect_solo(parent_agent, prompt, orch_cfg, guidance, m)
    if strategy == ARCHITECT_PLAN:
        return _plan_implement_review(parent_agent, prompt, orch_cfg, guidance, m)
    if strategy == WORKER_IMPLEMENT_REVIEW:
        return _implement_review(parent_agent, prompt, orch_cfg, guidance, m)
    # WORKER_SOLO
    review_enabled = bool((orch_cfg.get("review") or {}).get("enabled", True))
    if decision.review_required and review_enabled:
        return _implement_review(parent_agent, prompt, orch_cfg, guidance, m)
    return _worker_solo(parent_agent, prompt, orch_cfg, guidance, m)


def _architect_solo(
    parent_agent: Any,
    prompt: str,
    orch_cfg: Dict[str, Any],
    guidance: Dict[str, str],
    m: Dict[str, Any],
) -> Dict[str, Any]:
    logger.info("[architect] model=%s", _role_model_log(orch_cfg, ROLE_ARCHITECT))
    goal = _architect_prompt(prompt, guidance)
    arch = _run_role(parent_agent, prompt, orch_cfg, ROLE_ARCHITECT, goal, m)
    if not arch.get("ok"):
        return {"ok": False, "final_response": arch.get("summary") or ""}
    return {"ok": True, "final_response": arch.get("summary") or ""}


def _plan_implement_review(
    parent_agent: Any,
    prompt: str,
    orch_cfg: Dict[str, Any],
    guidance: Dict[str, str],
    m: Dict[str, Any],
) -> Dict[str, Any]:
    logger.info("[architect] model=%s", _role_model_log(orch_cfg, ROLE_ARCHITECT))
    arch_goal = _architect_prompt(prompt, guidance)
    arch = _run_role(parent_agent, prompt, orch_cfg, ROLE_ARCHITECT, arch_goal, m)
    if not arch.get("ok"):
        return {"ok": False, "final_response": arch.get("summary") or ""}
    brief = arch.get("summary") or ""
    acceptance = _extract_acceptance_criteria(brief)
    return _implement_review(
        parent_agent,
        prompt,
        orch_cfg,
        guidance,
        m,
        architect_brief=brief,
        acceptance_criteria=acceptance,
    )


def _implement_review(
    parent_agent: Any,
    prompt: str,
    orch_cfg: Dict[str, Any],
    guidance: Dict[str, str],
    m: Dict[str, Any],
    architect_brief: Optional[str] = None,
    acceptance_criteria: Optional[List[str]] = None,
) -> Dict[str, Any]:
    acceptance_criteria = acceptance_criteria or []
    review_cfg = orch_cfg.get("review") or {}
    review_enabled = bool(review_cfg.get("enabled", True))
    max_iterations = int(review_cfg.get("max_iterations", 2) or 0)
    escalation_cfg = orch_cfg.get("escalation") or {}
    escalation_enabled = bool(escalation_cfg.get("enabled", True))
    max_worker_failures = int(escalation_cfg.get("max_worker_failures", 3) or 0)

    revision = 0
    revision_context: Optional[str] = None
    worker_failures = 0

    while True:
        goal = _worker_prompt(
            prompt,
            guidance,
            architect_brief=architect_brief,
            acceptance_criteria=acceptance_criteria,
            revision_context=revision_context,
        )
        logger.info("[worker] model=%s", _role_model_log(orch_cfg, ROLE_WORKER))
        worker = _run_role(parent_agent, prompt, orch_cfg, ROLE_WORKER, goal, m)

        if not worker.get("ok"):
            worker_failures += 1
            m["worker_failures"] = worker_failures
            if escalation_enabled and worker_failures >= max_worker_failures:
                logger.warning(
                    "[escalation] worker failed %d times; escalating", worker_failures
                )
                m["escalations"] = m.get("escalations", 0) + 1
                return _escalate(
                    parent_agent, prompt, orch_cfg, guidance, m, worker
                )
            continue

        if not review_enabled:
            return _accepted(worker)

        reviewer_goal = _reviewer_prompt(
            prompt,
            guidance,
            acceptance_criteria,
            worker,
        )
        logger.info("[reviewer] model=%s", _role_model_log(orch_cfg, ROLE_REVIEWER))
        review = _run_role(parent_agent, prompt, orch_cfg, ROLE_REVIEWER, reviewer_goal, m)
        verdict, issues, required, confidence = _parse_verdict(review)
        m["verdicts"].append(verdict)
        logger.info(
            "[reviewer] verdict=%s confidence=%s", verdict, confidence
        )

        if verdict == VERDICT_ACCEPT:
            return _accepted(worker)

        if verdict == VERDICT_ESCALATE or revision >= max_iterations:
            logger.info(
                "[escalation] verdict=%s revision=%d max=%d; escalating",
                verdict,
                revision,
                max_iterations,
            )
            m["escalations"] = m.get("escalations", 0) + 1
            if escalation_enabled:
                return _escalate(
                    parent_agent, prompt, orch_cfg, guidance, m, worker,
                    issues=issues, required=required,
                )
            return _failed(
                "review could not reach acceptance within the revision budget"
            )

        # verdict == revise
        m["review_iterations"] = revision + 1
        revision += 1
        logger.info("[worker] revision=%d", revision)
        revision_context = _format_revision_context(issues, required)


def _worker_solo(
    parent_agent: Any,
    prompt: str,
    orch_cfg: Dict[str, Any],
    guidance: Dict[str, str],
    m: Dict[str, Any],
) -> Dict[str, Any]:
    goal = _worker_prompt(prompt, guidance)
    logger.info("[worker] model=%s", _role_model_log(orch_cfg, ROLE_WORKER))
    worker = _run_role(parent_agent, prompt, orch_cfg, ROLE_WORKER, goal, m)
    return _accepted(worker) if worker.get("ok") else _failed(worker.get("summary") or "")


def _escalate(
    parent_agent: Any,
    prompt: str,
    orch_cfg: Dict[str, Any],
    guidance: Dict[str, str],
    m: Dict[str, Any],
    worker: Dict[str, Any],
    issues: Optional[List[str]] = None,
    required: Optional[List[str]] = None,
) -> Dict[str, Any]:
    depth = int(m.get("_escalation_depth") or 0)
    if depth < 1:
        m["_escalation_depth"] = depth + 1
        logger.info("[escalation] escalating to architect (depth=%d)", depth)
        ctx = _format_escalation_context(prompt, worker, issues, required)
        arch_goal = _architect_prompt(prompt + "\n\nESCALATION CONTEXT:\n" + ctx, guidance)
        arch = _run_role(parent_agent, prompt, orch_cfg, ROLE_ARCHITECT, arch_goal, m)
        if arch.get("ok"):
            brief = arch.get("summary") or ""
            acceptance = _extract_acceptance_criteria(brief)
            return _implement_review(
                parent_agent, prompt, orch_cfg, guidance, m,
                architect_brief=brief, acceptance_criteria=acceptance,
            )
    return _failed("orchestration escalated and could not recover")


def _accepted(worker: Dict[str, Any]) -> Dict[str, Any]:
    return {"ok": True, "final_response": worker.get("summary") or ""}


def _failed(message: str) -> Dict[str, Any]:
    return {"ok": False, "final_response": f"[orchestration failed] {message}"}


# ---------------------------------------------------------------------------
# Prompt building (skill-driven with built-in fallback)
# ---------------------------------------------------------------------------


def _architect_prompt(prompt: str, guidance: Dict[str, str]) -> str:
    md = guidance.get("markdown")
    if md:
        return (
            "You are the architect in a multi-model orchestration. "
            "Follow the guidance below.\n\n"
            "--- SKILL GUIDANCE ---\n"
            f"{md}\n"
            "--- END SKILL GUIDANCE ---\n\n"
            "ORIGINAL REQUEST (authoritative):\n"
            f"{prompt}\n\n"
            "Produce a COMPACT structured implementation brief with sections: "
            "objective, constraints, files_components_likely_affected, "
            "implementation_approach, acceptance_criteria, risks, tests_required. "
            "Do NOT write a long essay or dump hidden chain-of-thought."
        )
    return (
        "You are the architect. Produce a COMPACT structured implementation "
        "brief (objective, constraints, files, approach, acceptance_criteria, "
        "risks, tests_required). Do not write a long essay.\n\n"
        f"ORIGINAL REQUEST:\n{prompt}\n"
    )


def _worker_prompt(
    prompt: str,
    guidance: Dict[str, str],
    architect_brief: Optional[str] = None,
    acceptance_criteria: Optional[List[str]] = None,
    revision_context: Optional[str] = None,
) -> str:
    md = guidance.get("markdown")
    lines = []
    if md:
        lines.append(
            "You are the implementing worker in a multi-model orchestration. "
            "Follow the guidance below."
        )
        lines.append("--- SKILL GUIDANCE ---")
        lines.append(md)
        lines.append("--- END SKILL GUIDANCE ---")
    else:
        lines.append(
            "You are the implementing worker. Complete the task with your normal "
            "tools and repository access. Report changed files, tests run and "
            "results, and any uncertainty."
        )
    lines.append("")
    lines.append("ORIGINAL REQUEST (authoritative):")
    lines.append(prompt)
    if architect_brief:
        lines += ["", "ARCHITECT BRIEF:", architect_brief]
    if acceptance_criteria:
        lines += ["", "ACCEPTANCE CRITERIA:"] + [
            f"  - {c}" for c in acceptance_criteria
        ]
    if revision_context:
        lines += ["", "REVISION FEEDBACK FROM REVIEWER:", revision_context]
    return "\n".join(lines)


def _reviewer_prompt(
    prompt: str,
    guidance: Dict[str, str],
    acceptance_criteria: List[str],
    worker: Dict[str, Any],
) -> str:
    md = guidance.get("markdown")
    tool_summary = _summarize_tool_trace(worker.get("tool_trace"))
    criteria = "\n".join(f"  - {c}" for c in acceptance_criteria) or "  (none)"
    lines = []
    if md:
        lines.append(
            "You are the independent reviewer in a multi-model orchestration. "
            "Follow the guidance below."
        )
        lines.append("--- SKILL GUIDANCE ---")
        lines.append(md)
        lines.append("--- END SKILL GUIDANCE ---")
    else:
        lines.append(
            "You are an independent reviewer. Review without continuing the "
            "worker's chain-of-thought. Inspect the repo yourself."
        )
    lines += [
        "",
        "ORIGINAL REQUEST:",
        prompt,
        "",
        f"ACCEPTANCE CRITERIA:\n{criteria}",
        "",
        f"WORKER SUMMARY:\n{worker.get('summary') or ''}",
        "",
        f"WORKER MODEL: {worker.get('model') or 'unknown'}",
        "",
        f"TOOL CALL TRACE:\n{tool_summary}",
        "",
        'Respond with ONLY JSON: {"verdict":"accept|revise|escalate",'
        '"issues":[],"required_changes":[],"confidence":0.0}',
    ]
    return "\n".join(lines)


def _summarize_tool_trace(trace) -> str:
    if not trace:
        return "(none)"
    out = []
    for item in trace[:40]:
        if isinstance(item, dict):
            status = item.get("status", "")
            out.append(f"  {item.get('tool','unknown')}" + (f" [{status}]" if status else ""))
    return "\n".join(out) or "(none)"


# ---------------------------------------------------------------------------
# Role execution (shared seam with Solution 1)
# ---------------------------------------------------------------------------


def _run_role(
    parent_agent: Any,
    prompt: str,
    orch_cfg: Dict[str, Any],
    role: str,
    goal: str,
    m: Dict[str, Any],
) -> Dict[str, Any]:
    role_cfg = resolve_role_model_config(orch_cfg, role)
    reasoning_config = build_role_reasoning_config(role_cfg)
    if reasoning_config and reasoning_config.get("enabled") is False:
        reasoning_config = None

    child_result = _build_and_run_child(
        parent_agent,
        goal,
        role=role,
        model=role_cfg["model"] or None,
        provider=role_cfg["provider"] or None,
        reasoning_config=reasoning_config,
        effort=role_cfg.get("reasoning_effort") or "",
    )
    normalized = _normalize_role_result(child_result, role)
    _record_role_metrics(m, role, normalized)
    return normalized


def _role_toolsets(role: str, effort: str) -> Optional[List[str]]:
    """Return a restricted toolset for a role child, or None to inherit parent.

    The worker child otherwise inherits the parent's full toolset and the
    default agent loop, which drives broad repo exploration (search_files,
    session_search, long terminal commands) even for trivial tasks. Restricting
    the toolset by role+effort keeps cheap tasks cheap and fast.

    - worker low: file only (read/write/patch/search) — no terminal, no web,
      no session_search. Enough for formatting / one-line edits / renames.
    - worker high/max: file + terminal (needed for tests, builds, debugging).
    - architect / reviewer: file + terminal (they inspect the repo and may run
      tests to verify).
    """
    if role == "worker":
        if effort in ("low", "minimal", "none"):
            return ["file"]
        return ["file", "terminal"]
    # architect / reviewer
    return ["file", "terminal"]


def _build_and_run_child(
    parent_agent: Any,
    goal: str,
    *,
    role: str,
    model: Optional[str],
    provider: Optional[str],
    reasoning_config: Optional[Dict[str, Any]],
    effort: str = "",
) -> Dict[str, Any]:
    """Build and run one role child via the existing delegate_tool primitive.

    Identical seam to Solution 1's ``orchestrator._build_and_run_child``.
    """
    from tools.delegate_tool import (
        _build_child_preserving_parent_tools,
        _resolve_delegation_credentials,
        _run_single_child,
        _load_config,
    )

    cfg = _load_config()
    try:
        creds = _resolve_delegation_credentials(cfg, parent_agent)
    except ValueError as exc:
        return {"status": "error", "error": str(exc), "summary": None}

    override_provider = provider
    override_base_url = creds.get("base_url")
    override_api_key = creds.get("api_key")
    override_api_mode = creds.get("api_mode")
    effective_model = model or creds.get("model")

    if provider:
        try:
            from hermes_cli.runtime_provider import resolve_runtime_provider

            runtime = resolve_runtime_provider(
                requested=provider,
                target_model=model or None,
            )
            if runtime:
                override_provider = runtime.get("provider") or provider
                override_base_url = runtime.get("base_url") or None
                override_api_key = runtime.get("api_key") or override_api_key
                override_api_mode = runtime.get("api_mode") or None
                effective_model = model or runtime.get("model") or effective_model
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[orchestrator/skill] failed to resolve role provider '%s': %s",
                provider,
                exc,
            )
            return {
                "status": "error",
                "error": f"cannot resolve role provider '{provider}': {exc}",
                "summary": None,
            }

    max_iterations = int(cfg.get("max_iterations") or 250)
    # Restrict the child's toolset by role+effort so cheap tasks don't trigger
    # broad repo exploration, and cap iterations tighter for low-effort work.
    toolsets = _role_toolsets(role, effort)
    if effort in ("low", "minimal", "none"):
        max_iterations = min(max_iterations, 8)
    try:
        child = _build_child_preserving_parent_tools(
            task_index=0,
            goal=goal,
            context=None,
            toolsets=toolsets,
            model=effective_model,
            max_iterations=max_iterations,
            task_count=1,
            parent_agent=parent_agent,
            override_provider=override_provider,
            override_base_url=override_base_url,
            override_api_key=override_api_key,
            override_api_mode=override_api_mode,
            role="leaf",
        )
        if reasoning_config is not None:
            try:
                child.reasoning_config = reasoning_config
            except Exception:
                pass
        result = _run_single_child(0, goal, child, parent_agent)
    except BaseException as exc:  # noqa: BLE001
        return {"status": "error", "error": str(exc), "summary": None}
    return result if isinstance(result, dict) else {"status": "error"}


def _normalize_role_result(child_result: Dict[str, Any], role: str) -> Dict[str, Any]:
    status = child_result.get("status")
    ok = status in ("completed", "success") and not child_result.get("failed")
    summary = child_result.get("summary")
    if isinstance(summary, (dict, list)):
        summary = json.dumps(summary, ensure_ascii=False)
    tokens = child_result.get("tokens") or {}
    return {
        "ok": bool(ok),
        "role": role,
        "summary": summary or "",
        "error": child_result.get("error"),
        "model": child_result.get("model"),
        "api_calls": int(child_result.get("api_calls") or 0),
        "duration_seconds": float(child_result.get("duration_seconds") or 0.0),
        "tokens": {
            "input": int(tokens.get("input") or 0),
            "output": int(tokens.get("output") or 0),
        },
        "tool_trace": child_result.get("tool_trace") or [],
        "cost_usd": float(child_result.get("cost_usd") or 0.0),
        "exit_reason": child_result.get("exit_reason"),
    }


def _record_role_metrics(m: Dict[str, Any], role: str, result: Dict[str, Any]) -> None:
    roles = m.setdefault("roles", {})
    entry = roles.get(role)
    if entry is None:
        entry = {
            "model": result.get("model"),
            "api_calls": 0,
            "tokens": {"input": 0, "output": 0},
            "duration_seconds": 0.0,
            "cost_usd": 0.0,
            "tool_calls": 0,
            "runs": 0,
        }
        roles[role] = entry
    entry["model"] = result.get("model") or entry["model"]
    entry["api_calls"] = entry["api_calls"] + int(result.get("api_calls") or 0)
    toks = result.get("tokens") or {}
    entry["tokens"]["input"] += int(toks.get("input") or 0)
    entry["tokens"]["output"] += int(toks.get("output") or 0)
    entry["duration_seconds"] = round(
        entry["duration_seconds"] + float(result.get("duration_seconds") or 0.0), 3
    )
    entry["cost_usd"] = round(
        entry["cost_usd"] + float(result.get("cost_usd") or 0.0), 6
    )
    entry["tool_calls"] += len(result.get("tool_trace") or [])
    entry["runs"] += 1

    m["total_api_calls"] = m.get("total_api_calls", 0) + int(result.get("api_calls") or 0)
    m["tool_calls"] = m.get("tool_calls", 0) + len(result.get("tool_trace") or [])
    m["total_tokens"]["input"] += int(toks.get("input") or 0)
    m["total_tokens"]["output"] += int(toks.get("output") or 0)


def _parse_verdict(reviewer_result: Dict[str, Any]) -> tuple:
    summary = reviewer_result.get("summary") or ""
    parsed = _extract_json_object(summary)
    if not isinstance(parsed, dict):
        return VERDICT_REVISE, [], [], 0.0
    verdict = str(parsed.get("verdict") or "").strip().lower()
    if verdict not in _VALID_VERDICTS:
        verdict = VERDICT_REVISE
    issues = parsed.get("issues") or []
    if not isinstance(issues, list):
        issues = []
    required = parsed.get("required_changes") or []
    if not isinstance(required, list):
        required = []
    try:
        confidence = float(parsed.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return verdict, [str(i) for i in issues], [str(r) for r in required], confidence


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    import re

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except (ValueError, TypeError):
        pass
    start = candidate.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(candidate)):
        ch = candidate[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(candidate[start : i + 1])
                    if isinstance(obj, dict):
                        return obj
                except (ValueError, TypeError):
                    return None
    return None


def _extract_acceptance_criteria(brief: str) -> List[str]:
    criteria = []
    if not brief:
        return criteria
    in_section = False
    for line in brief.splitlines():
        low = line.strip().lower()
        if low.startswith("acceptance_criteria") or low.startswith("acceptance criteria"):
            in_section = True
            continue
        if in_section:
            if not line.strip():
                continue
            if any(
                low.startswith(p)
                for p in (
                    "objective",
                    "constraints",
                    "files_components",
                    "implementation_approach",
                    "risks",
                    "tests_required",
                )
            ):
                in_section = False
                break
            criteria.append(line.strip().lstrip("-* "))
    return criteria


def _format_revision_context(issues, required) -> str:
    parts = []
    if issues:
        parts.append("Issues:\n" + "\n".join(f"- {i}" for i in issues))
    if required:
        parts.append("Required changes:\n" + "\n".join(f"- {c}" for c in required))
    return "\n\n".join(parts) or "Reviewer requested changes."


def _format_escalation_context(prompt, worker, issues, required) -> str:
    lines = [f"Original request: {prompt}"]
    if issues:
        lines.append("Issues:\n" + "\n".join(f"- {i}" for i in issues))
    if required:
        lines.append("Required changes:\n" + "\n".join(f"- {c}" for c in required))
    if worker.get("error"):
        lines.append(f"Worker error: {worker['error']}")
    return "\n".join(lines)


def _role_model_log(orch_cfg: Dict[str, Any], role: str) -> str:
    cfg = resolve_role_model_config(orch_cfg, role)
    return cfg.get("model") or "inherit-parent"
