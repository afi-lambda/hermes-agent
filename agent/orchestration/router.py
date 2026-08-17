"""
Deterministic, configurable heuristic router.

Maps a task description to a routing strategy + worker reasoning effort +
review flag using only task dimensions (complexity, ambiguity, risk, scope,
expected_tool_usage, need_for_architecture, need_for_independent_review).

This is intentionally NOT an ML classifier. It is keyword/dimension heuristics
plus an explicit config override (routing.mode: manual -> honor the configured
``strategy``), so the five benchmark modes can be forced without code changes.
Model-assisted routing is a documented future hook, not implemented here.
"""

from __future__ import annotations

from typing import Any, Dict, List

from agent.orchestration.config import (
    DEFAULT_STRATEGY,
    STRATEGIES,
)

# Strategy name constants (generic, model-independent).
WORKER_SOLO = "WORKER_SOLO"
ARCHITECT_SOLO = "ARCHITECT_SOLO"
ARCHITECT_PLAN = "ARCHITECT_PLAN_WORKER_IMPLEMENT_REVIEW"
WORKER_IMPLEMENT_REVIEW = "WORKER_IMPLEMENT_REVIEW"

VALID_EFFORTS = ("low", "medium", "high", "max")


class RoutingDecision:
    """Structured output of the router (see spec section 4)."""

    __slots__ = (
        "strategy",
        "worker_reasoning_effort",
        "review_required",
        "reason",
    )

    def __init__(
        self,
        strategy: str,
        worker_reasoning_effort: str,
        review_required: bool,
        reason: str,
    ) -> None:
        self.strategy = strategy
        self.worker_reasoning_effort = worker_reasoning_effort
        self.review_required = review_required
        self.reason = reason

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable shape for logs/events."""
        return {
            "strategy": self.strategy,
            "worker_reasoning_effort": self.worker_reasoning_effort,
            "review_required": self.review_required,
            "reason": self.reason,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"RoutingDecision(strategy={self.strategy}, "
            f"effort={self.worker_reasoning_effort}, "
            f"review={self.review_required})"
        )


# Keyword sets for the dimension heuristics. Kept conservative: a task that
# matches none of the strong signals falls through to the default "ordinary
# coding" path.
_TINY_KEYWORDS = (
    "format",
    "formatting",
    "rename",
    "rename?",
    "one-line",
    "one line",
    "boilerplate",
    "typo",
    "lint",
    "reformat",
    "trivial",
    "small fix",
)

_ARCHITECTURAL_KEYWORDS = (
    "architect",
    "architecture",
    "ambiguous",
    "unclear",
    "high-risk",
    "high risk",
    "multi-file",
    "multi file",
    "refactor",
    "redesign",
    "design a",
    "design the",
    "spec",
    "plan",
    "scalable",
)

_DEBUG_KEYWORDS = (
    "debug",
    "debugging",
    "root cause",
    "fails",
    "failure",
    "hang",
    "crashes",
    "intermittent",
    "regression",
    "flaky",
    "why is",
    "not working",
)

_RISK_KEYWORDS = (
    "risky",
    "high-risk",
    "production",
    "deployment",
    "migration",
    "security",
    "irreversible",
    "destructive",
    "breaking change",
    "api change",
)


def _contains_any(text: str, keywords: tuple) -> bool:
    lowered = text.lower()
    for kw in keywords:
        if kw in lowered:
            return True
    return False


def _score_dimensions(task: str) -> Dict[str, int]:
    """Score task dimensions (0-5 each) from the task text.

    These feed the deterministic rules. They are simple term-count heuristics,
    not a model.
    """
    dims = {
        "complexity": 2,  # default moderate
        "ambiguity": 1,
        "risk": 0,
        "scope": 1,
        "expected_tool_usage": 1,
        "need_for_architecture": 0,
        "need_for_independent_review": 0,
    }
    text = (task or "").lower()
    words = text.split()

    if _contains_any(text, _TINY_KEYWORDS):
        dims["complexity"] = 1
        dims["scope"] = 1
    if _contains_any(text, _ARCHITECTURAL_KEYWORDS):
        dims["need_for_architecture"] = 5
        dims["complexity"] = max(dims["complexity"], 4)
        dims["scope"] = max(dims["scope"], 3)
        dims["ambiguity"] = max(dims["ambiguity"], 3)
    if _contains_any(text, _DEBUG_KEYWORDS):
        dims["complexity"] = max(dims["complexity"], 4)
        dims["need_for_independent_review"] = max(
            dims["need_for_independent_review"], 2
        )
    if _contains_any(text, _RISK_KEYWORDS):
        dims["risk"] = 4
        dims["need_for_independent_review"] = max(
            dims["need_for_independent_review"], 3
        )

    # Long / multi-file tasks hint at broader scope.
    if len(words) > 80:
        dims["complexity"] = max(dims["complexity"], 3)
    if "multiple files" in text or "across the" in text:
        dims["scope"] = max(dims["scope"], 4)
        dims["complexity"] = max(dims["complexity"], 3)

    return dims


def route(task: str, config: Dict[str, Any]) -> RoutingDecision:
    """Return a routing decision for ``task``.

    ``config`` is the normalized orchestration config from
    ``load_orchestration_config``.
    """
    routing = config.get("routing") or {}
    mode = str(routing.get("mode") or "heuristic").strip().lower()

    # Manual override (for benchmarking the 5 modes) wins over heuristics.
    if mode == "manual":
        strategy = str(config.get("strategy") or "").strip().upper()
        if strategy in STRATEGIES:
            if strategy == ARCHITECT_PLAN:
                return RoutingDecision(
                    strategy=strategy,
                    worker_reasoning_effort="high",
                    review_required=True,
                    reason="manual strategy override",
                )
            if strategy == WORKER_IMPLEMENT_REVIEW:
                return RoutingDecision(
                    strategy=strategy,
                    worker_reasoning_effort="high",
                    review_required=True,
                    reason="manual strategy override",
                )
            if strategy == ARCHITECT_SOLO:
                return RoutingDecision(
                    strategy=strategy,
                    worker_reasoning_effort="high",
                    review_required=False,
                    reason="manual strategy override",
                )
            # WORKER_SOLO
            return RoutingDecision(
                strategy=strategy,
                worker_reasoning_effort="high",
                review_required=False,
                reason="manual strategy override",
            )
        # Invalid manual strategy -> fall through to heuristics.

    dims = _score_dimensions(task)

    # Architectural / ambiguous / high-risk -> architect plan -> worker -> reviewer.
    if (
        dims["need_for_architecture"] >= 4
        or dims["risk"] >= 4
        or dims["ambiguity"] >= 4
    ):
        return RoutingDecision(
            strategy=ARCHITECT_PLAN,
            worker_reasoning_effort="max",
            review_required=True,
            reason=(
                "architectural/ambiguous/high-risk task: "
                f"architecture={dims['need_for_architecture']} "
                f"risk={dims['risk']} ambiguity={dims['ambiguity']}"
            ),
        )

    # Very small / mechanical -> worker low, no review.
    if dims["complexity"] <= 1 and dims["scope"] <= 1 and dims["risk"] == 0:
        return RoutingDecision(
            strategy=WORKER_SOLO,
            worker_reasoning_effort="low",
            review_required=False,
            reason="very small / mechanical task (low complexity, low risk)",
        )

    # Difficult debugging -> worker max, review if risk elevated.
    if dims["complexity"] >= 4 and dims["need_for_independent_review"] >= 2:
        return RoutingDecision(
            strategy=(
                WORKER_IMPLEMENT_REVIEW
                if dims["need_for_independent_review"] >= 3
                else WORKER_SOLO
            ),
            worker_reasoning_effort="max",
            review_required=dims["need_for_independent_review"] >= 3,
            reason=(
                "difficult debugging: complexity={} review_need={}".format(
                    dims["complexity"], dims["need_for_independent_review"]
                )
            ),
        )

    # Ordinary coding task -> worker high, optional review if risk elevated.
    review_required = dims["risk"] >= 3 or dims["need_for_independent_review"] >= 3
    strategy = WORKER_IMPLEMENT_REVIEW if review_required else WORKER_SOLO
    return RoutingDecision(
        strategy=strategy,
        worker_reasoning_effort="high",
        review_required=review_required,
        reason=(
            "ordinary coding task: complexity={} risk={} review={}".format(
                dims["complexity"], dims["risk"], review_required
            )
        ),
    )


def default_decision() -> RoutingDecision:
    """Fallback used when routing fails or a default is required."""
    return RoutingDecision(
        strategy=DEFAULT_STRATEGY,
        worker_reasoning_effort="high",
        review_required=False,
        reason="default routing fallback",
    )
