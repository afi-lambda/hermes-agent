#!/usr/bin/env python3
"""
Tests for the deterministic orchestration router.

Covers spec sections 4 & 5 routing policy: tiny -> worker low, ordinary coding
-> worker high, difficult -> worker max (or equivalent), architectural ->
architect -> worker -> reviewer, and routing config override (manual mode).
"""

import unittest

from agent.orchestration.config import load_orchestration_config
from agent.orchestration.router import (
    ARCHITECT_PLAN,
    ARCHITECT_SOLO,
    WORKER_IMPLEMENT_REVIEW,
    WORKER_SOLO,
    RoutingDecision,
    route,
)

# A config with default heuristics + one role-pinned for realism.
BASE_CFG = load_orchestration_config(
    {
        "orchestration": {
            "enabled": True,
            "models": {
                "worker": {"model": "deepseek-v4-flash", "reasoning_effort": "high"},
                "architect": {"model": "glm-5.2"},
                "reviewer": {"model": "glm-5.2"},
            },
        }
    }
)


def _route(text, config=BASE_CFG):
    return route(text, config)


class TestRouterTinyTask(unittest.TestCase):
    def test_formatting_routes_worker_low(self):
        decision = _route("Reformat this Python file with ruff")
        self.assertEqual(decision.strategy, WORKER_SOLO)
        self.assertEqual(decision.worker_reasoning_effort, "low")
        self.assertFalse(decision.review_required)

    def test_rename_routes_worker_low(self):
        decision = _route("Rename the function `foo` to `bar`")
        self.assertEqual(decision.strategy, WORKER_SOLO)
        self.assertEqual(decision.worker_reasoning_effort, "low")

    def test_one_line_fix_routes_worker_low(self):
        decision = _route("Fix the typo on line 12")
        self.assertEqual(decision.worker_reasoning_effort, "low")


class TestRouterOrdinaryCoding(unittest.TestCase):
    def test_bounded_feature_routes_worker_high(self):
        decision = _route("Add a new config option to enable verbose logging")
        self.assertEqual(decision.strategy, WORKER_SOLO)
        self.assertEqual(decision.worker_reasoning_effort, "high")
        self.assertFalse(decision.review_required)

    def test_test_creation_routes_worker_high(self):
        decision = _route("Write unit tests for the new helper function")
        self.assertEqual(decision.worker_reasoning_effort, "high")


class TestRouterDifficultDebug(unittest.TestCase):
    def test_debug_task_routes_worker_max(self):
        decision = _route(
            "The test suite is flaky and fails intermittently; debug the "
            "root cause of the intermittent CI failure"
        )
        self.assertEqual(decision.worker_reasoning_effort, "max")

    def test_high_risk_coding_adds_review(self):
        # "high-risk" / "breaking change" / "production" -> risk>=4, which the
        # router treats as architect-plan (spec: high-risk -> plan+review).
        decision = _route(
            "Deploy a breaking change to the production API; this is high-risk"
        )
        self.assertEqual(decision.strategy, ARCHITECT_PLAN)
        self.assertTrue(decision.review_required)

    def test_elevated_risk_ordinary_coding_adds_review(self):
        # "migration" is a risk keyword -> routes to architect-plan + review.
        decision = _route(
            "Add an API migration that could affect existing callers"
        )
        self.assertEqual(decision.strategy, ARCHITECT_PLAN)
        self.assertTrue(decision.review_required)

    def test_review_needed_for_regression_risk(self):
        # Debug need at level 2 -> worker max; no review yet (escalation later).
        decision = _route(
            "Debug a regression that broke the build across multiple files"
        )
        self.assertEqual(decision.worker_reasoning_effort, "max")
        self.assertFalse(decision.review_required)

    def test_debug_plus_risk_routes_to_review(self):
        # Debug + risk keyword -> review_need>=3 -> worker max + review.
        decision = _route(
            "Debug a security regression that is high-risk before deployment"
        )
        self.assertEqual(decision.worker_reasoning_effort, "max")
        self.assertTrue(decision.review_required)


class TestRouterArchitectural(unittest.TestCase):
    def test_architectural_task_routes_plan_implement_review(self):
        decision = _route(
            "Design the architecture for a new multi-file feature with an "
            "unclear and ambiguous spec; this is high-risk and involves "
            "multiple files"
        )
        self.assertEqual(decision.strategy, ARCHITECT_PLAN)
        self.assertTrue(decision.review_required)
        self.assertEqual(decision.worker_reasoning_effort, "max")

    def test_ambiguous_task_routes_architect(self):
        decision = _route(
            "The requirements are ambiguous and unclear; plan and architect "
            "the approach first"
        )
        self.assertEqual(decision.strategy, ARCHITECT_PLAN)


class TestRoutingOverride(unittest.TestCase):
    def test_manual_strategy_override_wins(self):
        manual_cfg = load_orchestration_config(
            {
                "orchestration": {
                    "enabled": True,
                    "routing": {"mode": "manual"},
                    "strategy": "WORKER_IMPLEMENT_REVIEW",
                }
            }
        )
        decision = route("Format this file with ruff", manual_cfg)
        self.assertEqual(decision.strategy, WORKER_IMPLEMENT_REVIEW)
        self.assertTrue(decision.review_required)

    def test_manual_architect_solo(self):
        manual_cfg = load_orchestration_config(
            {
                "orchestration": {
                    "enabled": True,
                    "routing": {"mode": "manual"},
                    "strategy": "ARCHITECT_SOLO",
                }
            }
        )
        decision = route("anything", manual_cfg)
        self.assertEqual(decision.strategy, ARCHITECT_SOLO)
        self.assertFalse(decision.review_required)

    def test_manual_plan_implement_review(self):
        manual_cfg = load_orchestration_config(
            {
                "orchestration": {
                    "enabled": True,
                    "routing": {"mode": "manual"},
                    "strategy": "ARCHITECT_PLAN_WORKER_IMPLEMENT_REVIEW",
                }
            }
        )
        decision = route("anything", manual_cfg)
        self.assertEqual(decision.strategy, ARCHITECT_PLAN)
        self.assertTrue(decision.review_required)


class TestRouterDecisionShape(unittest.TestCase):
    def test_to_dict_shape(self):
        decision = RoutingDecision(
            strategy=WORKER_SOLO,
            worker_reasoning_effort="high",
            review_required=False,
            reason="test",
        )
        d = decision.to_dict()
        self.assertEqual(
            set(d.keys()),
            {
                "strategy",
                "worker_reasoning_effort",
                "review_required",
                "reason",
            },
        )
        self.assertIsInstance(d["reason"], str)


if __name__ == "__main__":
    unittest.main()
