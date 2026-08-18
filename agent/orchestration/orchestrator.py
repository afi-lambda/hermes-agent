"""
Strategy driver for the multi-model orchestration layer.

Runs a chosen strategy by building/running one child agent per generic role
(worker / architect / reviewer), enforces the bounded revision loop, escalates
on failure, and collects structured metrics for later benchmarking.

All role children are built with Hermes's existing sub-agent primitive
(``tools.delegate_tool._build_child_agent``) using the per-role provider:model +
reasoning config. This reuses the agent framework rather than building a
parallel one.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from agent.orchestration.config import (
    resolve_role_model_config,
    build_role_reasoning_config,
)
from agent.orchestration.prompts import (
    architect_prompt,
    reviewer_prompt,
    worker_prompt,
)
from agent.orchestration.router import (
    ARCHITECT_PLAN,
    ARCHITECT_SOLO,
    WORKER_IMPLEMENT_REVIEW,
    WORKER_SOLO,
    RoutingDecision,
    route,
)

logger = logging.getLogger(__name__)

# Role names used in metrics/logs.
ROLE_WORKER = "worker"
ROLE_ARCHITECT = "architect"
ROLE_REVIEWER = "reviewer"

# Reviewer verdicts.
VERDICT_ACCEPT = "accept"
VERDICT_REVISE = "revise"
VERDICT_ESCALATE = "escalate"
_VALID_VERDICTS = {VERDICT_ACCEPT, VERDICT_REVISE, VERDICT_ESCALATE}


def run_orchestrated(
    parent_agent: Any,
    prompt: str,
    orch_cfg: Dict[str, Any],
    decision: Optional[RoutingDecision] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the orchestrator for ``prompt`` and return a run_conversation-shaped dict.

    ``decision`` may be pre-computed (by the router) or computed here from
    ``orch_cfg``. The returned dict carries ``final_response`` (the user-facing
    result), ``api_calls``, ``completed``/``failed`` flags, and an
    ``orchestration`` block with structured metrics.
    """
    if metrics is None:
        metrics = {}
    start = time.monotonic()

    if decision is None:
        decision = route(prompt, orch_cfg)

    orchestrator_metrics: Dict[str, Any] = {
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
    }

    logger.info(
        "[orchestrator] strategy=%s worker_effort=%s review=%s",
        decision.strategy,
        decision.worker_reasoning_effort,
        decision.review_required,
    )

    try:
        result = _dispatch_strategy(
            parent_agent,
            prompt,
            orch_cfg,
            decision,
            orchestrator_metrics,
        )
    except Exception as exc:  # noqa: BLE001 - orchestration must never kill the turn
        logger.warning("[orchestrator] fatal orchestration error: %s", exc)
        result = _failure_result(
            prompt,
            error=f"orchestration failed: {exc}",
            metrics=orchestrator_metrics,
        )

    orchestrator_metrics["wall_clock_seconds"] = round(
        time.monotonic() - start, 3
    )

    final_response = _extract_final_response(result)
    metrics.update(orchestrator_metrics)

    return {
        "final_response": final_response,
        "messages": [],
        "api_calls": orchestrator_metrics["total_api_calls"],
        "completed": bool(result.get("ok")),
        "failed": not bool(result.get("ok")),
        "orchestration": orchestrator_metrics,
    }


# ---------------------------------------------------------------------------
# Strategy dispatch
# ---------------------------------------------------------------------------


def _dispatch_strategy(
    parent_agent: Any,
    prompt: str,
    orch_cfg: Dict[str, Any],
    decision: RoutingDecision,
    m: Dict[str, Any],
) -> Dict[str, Any]:
    strategy = decision.strategy
    if strategy == ARCHITECT_SOLO:
        return _run_architect_solo(parent_agent, prompt, orch_cfg, m)
    if strategy == ARCHITECT_PLAN:
        return _run_plan_implement_review(parent_agent, prompt, orch_cfg, m)
    if strategy == WORKER_IMPLEMENT_REVIEW:
        return _run_implement_review(parent_agent, prompt, orch_cfg, m)
    # WORKER_SOLO (and fallback)
    return _run_worker_solo(parent_agent, prompt, orch_cfg, m, decision)


def _run_worker_solo(
    parent_agent: Any,
    prompt: str,
    orch_cfg: Dict[str, Any],
    m: Dict[str, Any],
    decision: RoutingDecision,
) -> Dict[str, Any]:
    review_enabled = bool((orch_cfg.get("review") or {}).get("enabled", True))
    if decision.review_required and review_enabled:
        return _run_implement_review(parent_agent, prompt, orch_cfg, m)
    worker_result = _run_role(
        parent_agent,
        prompt,
        orch_cfg,
        ROLE_WORKER,
        goal=worker_prompt(prompt),
        m=m,
    )
    return _worker_result(worker_result)


def _run_architect_solo(
    parent_agent: Any,
    prompt: str,
    orch_cfg: Dict[str, Any],
    m: Dict[str, Any],
) -> Dict[str, Any]:
    logger.info("[architect] model=%s", _role_model_log(orch_cfg, ROLE_ARCHITECT))
    arch_result = _run_role(
        parent_agent,
        prompt,
        orch_cfg,
        ROLE_ARCHITECT,
        goal=architect_prompt(prompt),
        m=m,
    )
    if not arch_result.get("ok"):
        return arch_result
    brief = arch_result.get("summary") or ""
    return {
        "ok": True,
        "final_response": brief,
        "architect_brief": brief,
    }


def _run_plan_implement_review(
    parent_agent: Any,
    prompt: str,
    orch_cfg: Dict[str, Any],
    m: Dict[str, Any],
) -> Dict[str, Any]:
    """architect plan -> worker implement -> reviewer (bounded revision loop)."""
    logger.info(
        "[architect] model=%s",
        _role_model_log(orch_cfg, ROLE_ARCHITECT),
    )
    arch_result = _run_role(
        parent_agent,
        prompt,
        orch_cfg,
        ROLE_ARCHITECT,
        goal=architect_prompt(prompt),
        m=m,
    )
    if not arch_result.get("ok"):
        return arch_result
    brief = arch_result.get("summary") or ""
    acceptance = _extract_acceptance_criteria(brief)

    return _implement_and_review(
        parent_agent,
        prompt,
        orch_cfg,
        m,
        architect_brief=brief,
        acceptance_criteria=acceptance,
    )


def _run_implement_review(
    parent_agent: Any,
    prompt: str,
    orch_cfg: Dict[str, Any],
    m: Dict[str, Any],
) -> Dict[str, Any]:
    """worker implement -> reviewer (bounded revision loop), no architect."""
    return _implement_and_review(
        parent_agent,
        prompt,
        orch_cfg,
        m,
        architect_brief=None,
        acceptance_criteria=(),
    )


def _implement_and_review(
    parent_agent: Any,
    prompt: str,
    orch_cfg: Dict[str, Any],
    m: Dict[str, Any],
    *,
    architect_brief: Optional[str],
    acceptance_criteria: List[str],
) -> Dict[str, Any]:
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
        goal = worker_prompt(
            prompt,
            architect_brief=architect_brief,
            acceptance_criteria=acceptance_criteria,
            revision_context=revision_context,
        )
        logger.info("[worker] model=%s", _role_model_log(orch_cfg, ROLE_WORKER))
        worker_result = _run_role(
            parent_agent,
            prompt,
            orch_cfg,
            ROLE_WORKER,
            goal=goal,
            m=m,
        )
        if not worker_result.get("ok"):
            worker_failures += 1
            m["worker_failures"] = worker_failures
            if escalation_enabled and worker_failures >= max_worker_failures:
                logger.warning(
                    "[escalation] worker failed %d times; escalating",
                    worker_failures,
                )
                m["escalations"] = m.get("escalations", 0) + 1
                return _escalate_failure(
                    parent_agent, prompt, orch_cfg, m, worker_result
                )
            # A single failure may be transient; retry the worker once unless
            # escalation already fired. Bounded by worker_failures count.
            if worker_failures >= max_worker_failures:
                m["escalations"] = m.get("escalations", 0) + 1
                return _failure_result(
                    prompt,
                    error="worker failed repeatedly",
                    metrics=m,
                )
            continue

        if not review_enabled:
            return _worker_result(worker_result)

        reviewer_result = _run_reviewer(
            parent_agent,
            prompt,
            orch_cfg,
            acceptance_criteria,
            worker_result,
            m,
        )
        verdict, issues, required_changes, confidence = _parse_reviewer_result(
            reviewer_result
        )
        m["verdicts"].append(verdict)
        logger.info(
            "[reviewer] model=%s verdict=%s confidence=%s",
            _role_model_log(orch_cfg, ROLE_REVIEWER),
            verdict,
            confidence,
        )

        if verdict == VERDICT_ACCEPT:
            return _accepted_result(worker_result, m)

        if verdict == VERDICT_ESCALATE or revision >= max_iterations:
            logger.info(
                "[escalation] verdict=%s revision=%d max=%d; escalating",
                verdict,
                revision,
                max_iterations,
            )
            m["escalations"] = m.get("escalations", 0) + 1
            if escalation_enabled:
                return _escalate_failure(
                    parent_agent,
                    prompt,
                    orch_cfg,
                    m,
                    worker_result,
                    issues=issues,
                    required_changes=required_changes,
                )
            # Escalation disabled: return a clear failure rather than loop.
            return _failure_result(
                prompt,
                error="review could not reach acceptance within the revision budget",
                metrics=m,
            )

        # verdict == revise
        m["review_iterations"] = revision + 1
        revision += 1
        logger.info("[worker] revision=%d", revision)
        revision_context = _format_revision_context(issues, required_changes)
        continue


# ---------------------------------------------------------------------------
# Role execution
# ---------------------------------------------------------------------------


def _run_role(
    parent_agent: Any,
    prompt: str,
    orch_cfg: Dict[str, Any],
    role: str,
    goal: str,
    m: Dict[str, Any],
) -> Dict[str, Any]:
    """Build and run one role child; return a normalized result dict.

    This is the single seam that tests mock to avoid real LLM calls.
    """
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


def _record_role_metrics(m: Dict[str, Any], role: str, result: Dict[str, Any]) -> None:
    """Fold a role's execution into the orchestrator metrics block."""
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

    m["total_api_calls"] = m.get("total_api_calls", 0) + int(
        result.get("api_calls") or 0
    )
    m["tool_calls"] = m.get("tool_calls", 0) + len(result.get("tool_trace") or [])
    m["total_tokens"]["input"] += int(toks.get("input") or 0)
    m["total_tokens"]["output"] += int(toks.get("output") or 0)


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
    """Build a child agent via delegate_tool and run it synchronously.

    Reuses Hermes's existing sub-agent primitive so we inherit toolset
    isolation, credential resolution, retry/error handling, and token/cost
    accounting without building a parallel agent framework.
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

    # When the role pins a provider, we need a matching credential bundle.
    # delegate_tool resolves provider:model through the runtime provider system
    # only when delegation.provider is set; for a role-pinned provider we
    # resolve via the same runtime provider resolution used by CLI startup.
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
                "[orchestrator] failed to resolve role provider '%s': %s",
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
        # Apply per-role reasoning config to the child.
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
    """Normalize a delegate_tool child result into a role result dict."""
    status = child_result.get("status")
    ok = status in ("completed", "success") and not child_result.get("failed")
    summary = child_result.get("summary")
    if isinstance(summary, (dict, list)):
        summary = json.dumps(summary, ensure_ascii=False)
    tokens = child_result.get("tokens") or {}
    normalized = {
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
    return normalized


def _run_reviewer(
    parent_agent: Any,
    prompt: str,
    orch_cfg: Dict[str, Any],
    acceptance_criteria: List[str],
    worker_result: Dict[str, Any],
    m: Dict[str, Any],
) -> Dict[str, Any]:
    logger.info(
        "[reviewer] model=%s",
        _role_model_log(orch_cfg, ROLE_REVIEWER),
    )
    review_goal = reviewer_prompt(
        prompt,
        acceptance_criteria,
        worker_summary=worker_result.get("summary") or "",
        worker_model=worker_result.get("model"),
        tool_trace=worker_result.get("tool_trace"),
        metrics=_reviewer_metrics(worker_result),
    )
    return _run_role(
        parent_agent,
        prompt,
        orch_cfg,
        ROLE_REVIEWER,
        goal=review_goal,
        m=m,
    )


# ---------------------------------------------------------------------------
# Reviewer verdict parsing
# ---------------------------------------------------------------------------


def _parse_reviewer_result(reviewer_result: Dict[str, Any]) -> tuple:
    """Parse the reviewer's verdict from its summary.

    Returns ``(verdict, issues, required_changes, confidence)``. Unparseable
    output is treated as ``revise`` (low confidence) so the revision loop can
    make progress or escalate.
    """
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
    confidence = max(0.0, min(1.0, confidence))
    return verdict, [str(i) for i in issues], [str(r) for r in required], confidence


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Robustly extract the first JSON object from ``text``."""
    if not text:
        return None
    # Strip markdown code fences.
    import re

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except (ValueError, TypeError):
        pass
    # Fall back to the first balanced {...} object.
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


# ---------------------------------------------------------------------------
# Escalation & result helpers
# ---------------------------------------------------------------------------


def _escalate_failure(
    parent_agent: Any,
    prompt: str,
    orch_cfg: Dict[str, Any],
    m: Dict[str, Any],
    worker_result: Dict[str, Any],
    issues: Optional[List[str]] = None,
    required_changes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Escalate to the architect model for a fresh plan or return failure.

    When the architect role is configured, produce a revised brief and hand it
    back to the worker for one more attempt. Otherwise return a clear failure.
    """
    arch_cfg = resolve_role_model_config(orch_cfg, ROLE_ARCHITECT)
    arch_model = arch_cfg.get("model")
    # Bound escalation depth so a persistent failure/rejection cannot recurse
    # indefinitely. After one architect re-plan, return a clear failure.
    depth = int(m.get("_escalation_depth") or 0)
    if arch_model and depth < 1:
        m["_escalation_depth"] = depth + 1
        logger.info(
            "[escalation] escalating to architect model=%s (depth=%d)",
            arch_model,
            depth,
        )
        escalation_context = _format_escalation_context(
            prompt, worker_result, issues, required_changes
        )
        arch_goal = architect_prompt(
            prompt + "\n\nESCALATION CONTEXT:\n" + escalation_context
        )
        arch_result = _run_role(
            parent_agent,
            prompt,
            orch_cfg,
            ROLE_ARCHITECT,
            goal=arch_goal,
            m=m,
        )
        if arch_result.get("ok"):
            brief = arch_result.get("summary") or ""
            acceptance = _extract_acceptance_criteria(brief)
            return _implement_and_review(
                parent_agent,
                prompt,
                orch_cfg,
                m,
                architect_brief=brief,
                acceptance_criteria=acceptance,
            )
    # No architect or architect failed: return a clear failure.
    return _failure_result(
        prompt,
        error="orchestration escalated and could not recover",
        metrics=m,
    )


def _failure_result(
    prompt: str,
    error: str,
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "ok": False,
        "final_response": f"[orchestration failed] {error}",
        "error": error,
    }


def _worker_result(worker_result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ok": bool(worker_result.get("ok")),
        "final_response": worker_result.get("summary") or "",
        "worker_result": worker_result,
    }


def _accepted_result(
    worker_result: Dict[str, Any],
    m: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "ok": True,
        "final_response": worker_result.get("summary") or "",
        "worker_result": worker_result,
    }


def _extract_final_response(result: Dict[str, Any]) -> str:
    resp = result.get("final_response") or ""
    return str(resp)


# ---------------------------------------------------------------------------
# Metrics / helpers
# ---------------------------------------------------------------------------


def _extract_acceptance_criteria(brief: str) -> List[str]:
    """Best-effort extract of acceptance criteria lines from an architect brief."""
    criteria: List[str] = []
    if not brief:
        return criteria
    in_section = False
    for line in brief.splitlines():
        stripped = line.strip()
        low = stripped.lower()
        if low.startswith("acceptance_criteria") or low.startswith("acceptance criteria"):
            in_section = True
            continue
        if in_section:
            if not stripped:
                continue
            if any(
                low.startswith(prefix)
                for prefix in (
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
            criteria.append(stripped.lstrip("-* "))
    return criteria


def _reviewer_metrics(worker_result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model": worker_result.get("model"),
        "api_calls": worker_result.get("api_calls"),
        "duration_seconds": worker_result.get("duration_seconds"),
        "tokens": worker_result.get("tokens"),
        "exit_reason": worker_result.get("exit_reason"),
    }


def _format_revision_context(
    issues: List[str], required_changes: List[str]
) -> str:
    parts = []
    if issues:
        parts.append("Issues:\n" + "\n".join(f"- {i}" for i in issues))
    if required_changes:
        parts.append(
            "Required changes:\n" + "\n".join(f"- {c}" for c in required_changes)
        )
    return "\n\n".join(parts) or "Reviewer requested changes; see review feedback."


def _format_escalation_context(
    prompt: str,
    worker_result: Dict[str, Any],
    issues: Optional[List[str]],
    required_changes: Optional[List[str]],
) -> str:
    lines = [f"Original request: {prompt}"]
    if issues:
        lines.append("Issues:\n" + "\n".join(f"- {i}" for i in issues))
    if required_changes:
        lines.append(
            "Required changes:\n"
            + "\n".join(f"- {c}" for c in required_changes)
        )
    if worker_result.get("error"):
        lines.append(f"Worker error: {worker_result['error']}")
    return "\n".join(lines)


def _role_model_log(orch_cfg: Dict[str, Any], role: str) -> str:
    cfg = resolve_role_model_config(orch_cfg, role)
    model = cfg.get("model")
    if model:
        return model
    return "inherit-parent"
