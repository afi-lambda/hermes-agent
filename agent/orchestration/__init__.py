"""
Multi-model orchestration layer (experimental, Sol-Advisor-style).

Routes a task to generic ``worker`` / ``architect`` / ``reviewer`` roles
(initially mapped to DeepSeek-V4-Flash / GLM-5.2 via config) using Hermes's
existing sub-agent primitive. Disabled by default; when disabled Hermes behaves
exactly as before.

Public surface:
- ``maybe_route_task(parent_agent, prompt, conversation_history)`` — entry called
  by ``run_agent.AIAgent.run_conversation`` when orchestration is enabled.
- ``orchestration_should_run(parent_agent, conversation_history)`` — the guarded
  no-op-when-disabled predicate the hook uses.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def orchestration_should_run(
    parent_agent: Any,
    conversation_history: Optional[List[Dict[str, Any]]],
) -> bool:
    """Return True only when orchestration should take over this turn.

    Guarantees backward compatibility:
    - False when orchestration is disabled in config.
    - False for subagents (a child must never re-orchestrate).
    - False for follow-up turns (only a fresh task turn is routed) so
      mid-conversation behavior is unchanged.
    """
    if getattr(parent_agent, "platform", None) == "subagent":
        return False

    try:
        from agent.orchestration.config import load_orchestration_config

        orch_cfg = load_orchestration_config()
    except Exception:  # noqa: BLE001
        return False
    if not orch_cfg.get("enabled"):
        return False

    # Fresh task turn only: no prior turns in the provided history and no
    # prior assistant turns in the agent's own session state.
    if conversation_history:
        return False
    session_messages = getattr(parent_agent, "_session_messages", None) or []
    for msg in session_messages:
        if isinstance(msg, dict) and msg.get("role") in ("user", "assistant"):
            return False
    return True


def maybe_route_task(
    parent_agent: Any,
    prompt: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run the orchestrated task and return a run_conversation-shaped dict.

    Safe to call even when orchestration is disabled (returns a pass-through
    marker so the caller falls back to the normal loop), but the intended usage
    is behind ``orchestration_should_run``.
    """
    try:
        from agent.orchestration.config import load_orchestration_config
        from agent.orchestration.router import route

        orch_cfg = load_orchestration_config()
        if not orch_cfg.get("enabled"):
            return {"orchestration_disabled": True}

        decision = route(prompt, orch_cfg)
        logger.info(
            "[orchestrator] routing decision: strategy=%s effort=%s review=%s",
            decision.strategy,
            decision.worker_reasoning_effort,
            decision.review_required,
        )
        if orch_cfg.get("engine") == "skill":
            from agent.orchestration.skill_engine import run_skill_orchestrated

            logger.info("[orchestrator] engine=skill")
            return run_skill_orchestrated(
                parent_agent, prompt, orch_cfg, decision=decision
            )
        logger.info("[orchestrator] engine=python")
        from agent.orchestration.orchestrator import run_orchestrated

        return run_orchestrated(parent_agent, prompt, orch_cfg, decision=decision)
    except Exception as exc:  # noqa: BLE001 - never break the normal loop
        logger.warning(
            "[orchestrator] orchestration failed; caller should fall back: %s", exc
        )
        return {"orchestration_disabled": True, "orchestration_error": str(exc)}
