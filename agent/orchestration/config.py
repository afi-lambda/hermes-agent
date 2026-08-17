"""
Multi-model orchestration configuration.

Loads the ``orchestration`` config block from Hermes config and validates it.
Disabled by default; when disabled Hermes behaves exactly as before (the
integration hook in ``run_agent.AIAgent.run_conversation`` short-circuits on
``enabled == False``).

Roles are generic and model-independent: ``worker`` (implements), ``architect``
(plans / escalates), ``reviewer`` (independent review). Each role resolves an
arbitrary provider:model pair plus an optional reasoning effort through the same
runtime-provider resolution used by CLI/gateway startup.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


# Role names supported by the orchestration layer. Kept as a plain tuple so the
# rest of the layer can iterate without importing an enum.
SUPPORTED_ROLES = ("worker", "architect", "reviewer")

# Valid reasoning effort levels accepted by Hermes (see
# hermes_constants.VALID_REASONING_EFFORTS). "max" and "high" are the levels the
# Sol-Advisor experiment cares about for DeepSeek-V4-Flash.
VALID_REASONING_EFFORTS = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
    "ultra",
)

# Strategy names (generic, model-independent). Configurable so the five
# benchmark modes (A..E in the spec) can be exercised without rewriting the
# orchestration layer.
STRATEGIES = (
    "WORKER_SOLO",
    "ARCHITECT_SOLO",
    "ARCHITECT_PLAN_WORKER_IMPLEMENT_REVIEW",
    "WORKER_IMPLEMENT_REVIEW",
)

DEFAULT_STRATEGY = "WORKER_SOLO"

DEFAULT_ORCHESTRATION: Dict[str, Any] = {
    "enabled": False,
    # engine: which implementation to run when enabled.
    #   "python" — all-Python (Solution 1): prompts are coded in prompts.py.
    #   "skill"  — thin Python core + skill file (Solution 2): behavioral
    #              guidance + prompt templates live in the
    #              multi-model-orchestration skill; Python only enforces the
    #              deterministic loop skeleton and collects metrics.
    "engine": "python",
    "routing": {
        "mode": "heuristic",  # heuristic | manual
    },
    "models": {
        "worker": {
            "model": "",
            "provider": "",
            "reasoning_effort": "high",
            "supports_reasoning_effort": True,  # provider capability
        },
        "architect": {
            "model": "",
            "provider": "",
            "reasoning_effort": "",
            "supports_reasoning_effort": True,
        },
        "reviewer": {
            "model": "",
            "provider": "",
            "reasoning_effort": "",
            "supports_reasoning_effort": True,
        },
    },
    "review": {
        "enabled": True,
        "max_iterations": 2,
    },
    "escalation": {
        "enabled": True,
        "max_worker_failures": 3,
    },
    "strategy": "",
}


def _deep_merge(base: Any, override: Any) -> Any:
    """Recursively merge ``override`` over ``base`` (dicts merge by key)."""
    if isinstance(base, dict) and isinstance(override, dict):
        out = dict(base)
        for key, value in override.items():
            if key in out:
                out[key] = _deep_merge(out[key], value)
            else:
                out[key] = value
        return out
    return override


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off"):
            return False
    return default


def _coerce_effort(value: Any) -> str:
    """Normalize a reasoning-effort value to a valid level or empty string."""
    if value is None:
        return ""
    if isinstance(value, bool):
        # YAML ``false`` means thinking disabled -> "none".
        return "none" if value is False else ""
    text = str(value).strip().lower()
    if text in VALID_REASONING_EFFORTS:
        return text
    return ""


def load_orchestration_config(full_config: Any = None) -> Dict[str, Any]:
    """Load the orchestration config, deep-merged over defaults.

    ``full_config`` may be the full Hermes config dict (e.g. from
    ``load_config_readonly()``). When omitted it is loaded from the shared
    readonly loader. The returned dict is a plain structure (never the live
    config) so callers may mutate / inspect freely.
    """
    if full_config is None:
        try:
            from hermes_cli.config import load_config_readonly

            full_config = load_config_readonly()
        except Exception:
            full_config = {}

    user = {}
    if isinstance(full_config, dict):
        user = full_config.get("orchestration") or {}
    merged = _deep_merge(DEFAULT_ORCHESTRATION, user)

    # Normalize / coerce values so downstream code does not re-parse types.
    merged["enabled"] = _coerce_bool(merged.get("enabled"), False)
    engine = str(merged.get("engine") or "python").strip().lower()
    merged["engine"] = engine if engine in ("python", "skill") else "python"
    routing = merged.get("routing") or {}
    routing["mode"] = str(routing.get("mode") or "heuristic").strip().lower()
    merged["routing"] = routing

    models = merged.get("models") or {}
    for role in SUPPORTED_ROLES:
        role_cfg = models.get(role) or {}
        if not isinstance(role_cfg, dict):
            role_cfg = {}
        role_cfg = dict(role_cfg)
        role_cfg["model"] = str(role_cfg.get("model") or "").strip()
        role_cfg["provider"] = str(role_cfg.get("provider") or "").strip()
        role_cfg["reasoning_effort"] = _coerce_effort(role_cfg.get("reasoning_effort"))
        models[role] = role_cfg
    merged["models"] = models

    review = merged.get("review") or {}
    merged["review"] = {
        "enabled": _coerce_bool(review.get("enabled"), True),
        "max_iterations": _coerce_nonneg_int(review.get("max_iterations"), 2),
    }

    escalation = merged.get("escalation") or {}
    merged["escalation"] = {
        "enabled": _coerce_bool(escalation.get("enabled"), True),
        "max_worker_failures": _coerce_nonneg_int(
            escalation.get("max_worker_failures"), 3
        ),
    }

    strategy = str(merged.get("strategy") or "").strip().upper()
    merged["strategy"] = strategy if strategy in STRATEGIES else DEFAULT_STRATEGY

    return merged


def _coerce_nonneg_int(value: Any, default: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return n if n >= 0 else default


def resolve_role_model_config(
    orch_cfg: Dict[str, Any], role: str
) -> Dict[str, Any]:
    """Return the normalized model config for ``role`` (worker/architect/reviewer).

    An empty ``model`` + ``provider`` means "inherit the parent agent's model /
    provider / credentials" — the same contract as ``delegation.model``.
    """
    if role not in SUPPORTED_ROLES:
        role = "worker"
    models = orch_cfg.get("models") or {}
    role_cfg = models.get(role) or {}
    if not isinstance(role_cfg, dict):
        role_cfg = {}
    effort = _coerce_effort(role_cfg.get("reasoning_effort"))
    supports = role_cfg.get("supports_reasoning_effort", True)
    if isinstance(supports, str):
        supports = supports.strip().lower() in ("true", "1", "yes", "on")
    else:
        supports = bool(supports)
    return {
        "model": str(role_cfg.get("model") or "").strip(),
        "provider": str(role_cfg.get("provider") or "").strip(),
        "reasoning_effort": effort,
        "supports_reasoning_effort": supports,
    }


def build_role_reasoning_config(role_model_cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Build the ``reasoning_config`` dict to pass to a role's child agent.

    Returns ``None`` when the provider/model does not support reasoning effort
    (so the caller sends no effort dial at all, matching the "do not assume all
    providers support reasoning_effort" requirement) or when no effort was
    configured (inherit parent's default).
    """
    if not role_model_cfg.get("supports_reasoning_effort", True):
        return None
    effort = str(role_model_cfg.get("reasoning_effort") or "").strip()
    if not effort:
        return None
    from hermes_constants import parse_reasoning_effort

    return parse_reasoning_effort(effort)
