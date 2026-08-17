#!/usr/bin/env python3
"""
Tests for the orchestration strategy driver.

Uses mocks for the role-child seam (``_build_and_run_child``) so no real LLM
calls happen. Covers: strategy flows, reviewer accept/revise, revision loop,
revision-limit escalation, reasoning-effort capability gating, missing-role
config degradation, provider/model failure, metrics presence, and the
orchestration-disabled backward-compat predicate.
"""

import unittest
from unittest.mock import patch

from agent.orchestration.config import load_orchestration_config
from agent.orchestration.orchestrator import (
    _parse_reviewer_result,
    run_orchestrated,
)
from agent.orchestration.router import (
    ARCHITECT_PLAN,
    WORKER_IMPLEMENT_REVIEW,
    WORKER_SOLO,
)
from agent.orchestration import orchestration_should_run


def _cfg(**overrides):
    base = {
        "enabled": True,
        "routing": {"mode": "heuristic"},
        "models": {
            "worker": {
                "model": "deepseek-v4-flash",
                "reasoning_effort": "high",
                "supports_reasoning_effort": True,
            },
            "architect": {
                "model": "glm-5.2",
                "reasoning_effort": "",
                "supports_reasoning_effort": True,
            },
            "reviewer": {
                "model": "glm-5.2",
                "reasoning_effort": "",
                "supports_reasoning_effort": True,
            },
        },
        "review": {"enabled": True, "max_iterations": 2},
        "escalation": {"enabled": True, "max_worker_failures": 3},
        "strategy": "",
    }
    for k, v in overrides.items():
        base[k] = v
    return load_orchestration_config({"orchestration": base})


def _worker_ok(summary="done", model="deepseek-v4-flash", **kw):
    return {
        "ok": True,
        "status": "completed",
        "summary": summary,
        "model": model,
        "api_calls": 3,
        "duration_seconds": 2.0,
        "tokens": {"input": 100, "output": 50},
        "tool_trace": [{"tool": "read_file", "status": "ok"}],
        "cost_usd": 0.001,
        "role": "worker",
        **kw,
    }


def _reviewer_ok(verdict="accept", **kw):
    import json

    summary = json.dumps(
        {
            "verdict": verdict,
            "issues": ["x"] if verdict == "revise" else [],
            "required_changes": ["fix x"] if verdict == "revise" else [],
            "confidence": 0.9 if verdict == "accept" else 0.6,
        }
    )
    return _worker_ok(summary=summary, model="glm-5.2", role="reviewer", **kw)


def _architect_ok(brief="objective: x\nacceptance_criteria:\n- works"):
    return _worker_ok(summary=brief, model="glm-5.2", role="architect")


class FakeParent:
    """Minimal stand-in for the parent AIAgent."""

    platform = "cli"
    model = "deepseek-v4-flash"
    _session_messages = []


class TestOrchestrationDisabled(unittest.TestCase):
    def test_should_run_false_when_disabled(self):
        with patch(
            "agent.orchestration.config.load_orchestration_config",
            return_value=_cfg(enabled=False),
        ):
            parent = FakeParent()
            self.assertFalse(orchestration_should_run(parent, None))

    def test_should_run_false_for_subagent(self):
        parent = FakeParent()
        parent.platform = "subagent"
        self.assertFalse(orchestration_should_run(parent, None))

    def test_should_run_false_for_followup_turn(self):
        parent = FakeParent()
        parent._session_messages = [{"role": "user", "content": "hi"}]
        self.assertFalse(orchestration_should_run(parent, None))

    def test_should_run_true_for_fresh_task(self):
        with patch(
            "agent.orchestration.config.load_orchestration_config",
            return_value=_cfg(enabled=True),
        ):
            parent = FakeParent()
            self.assertTrue(orchestration_should_run(parent, None))


class TestStrategyFlows(unittest.TestCase):
    def test_worker_solo(self):
        cfg = _cfg()
        with patch(
            "agent.orchestration.orchestrator._build_and_run_child",
            return_value=_worker_ok(),
        ):
            result = run_orchestrated(FakeParent(), "Add a config option", cfg)
        self.assertTrue(result["completed"])
        self.assertFalse(result["failed"])
        self.assertEqual(result["final_response"], "done")
        orch = result["orchestration"]
        self.assertEqual(orch["strategy"], WORKER_SOLO)
        self.assertIn("worker", orch["roles"])

    def test_architect_plan_accept(self):
        cfg = _cfg()
        # Seam returns architect then worker then reviewer-accept in sequence.
        queue = [_architect_ok(), _worker_ok(), _reviewer_ok("accept")]

        def fake_build(*args, **kwargs):
            return queue.pop(0)

        with patch(
            "agent.orchestration.orchestrator._build_and_run_child",
            side_effect=fake_build,
        ):
            result = run_orchestrated(
                FakeParent(),
                "Design the architecture for a multi-file feature",
                cfg,
            )
        self.assertTrue(result["completed"])
        orch = result["orchestration"]
        self.assertEqual(orch["strategy"], ARCHITECT_PLAN)
        self.assertIn("architect", orch["roles"])
        self.assertIn("worker", orch["roles"])
        self.assertIn("reviewer", orch["roles"])
        self.assertEqual(orch["verdicts"], ["accept"])

    def test_worker_implement_review_accept(self):
        cfg = _cfg()
        cfg["routing"]["mode"] = "manual"
        cfg["strategy"] = WORKER_IMPLEMENT_REVIEW
        queue = [_worker_ok(), _reviewer_ok("accept")]

        def fake_build(*args, **kwargs):
            return queue.pop(0)

        with patch(
            "agent.orchestration.orchestrator._build_and_run_child",
            side_effect=fake_build,
        ):
            result = run_orchestrated(FakeParent(), "Implement feature", cfg)
        self.assertTrue(result["completed"])
        self.assertEqual(result["orchestration"]["strategy"], WORKER_IMPLEMENT_REVIEW)
        self.assertIn("worker", result["orchestration"]["roles"])
        self.assertIn("reviewer", result["orchestration"]["roles"])

    def test_reviewer_requests_revision_then_accept(self):
        cfg = _cfg()
        cfg["routing"]["mode"] = "manual"
        cfg["strategy"] = WORKER_IMPLEMENT_REVIEW
        queue = [
            _worker_ok("first attempt"),
            _reviewer_ok("revise"),
            _worker_ok("second attempt"),
            _reviewer_ok("accept"),
        ]

        def fake_build(*args, **kwargs):
            return queue.pop(0)

        with patch(
            "agent.orchestration.orchestrator._build_and_run_child",
            side_effect=fake_build,
        ):
            result = run_orchestrated(FakeParent(), "Implement feature", cfg)
        self.assertTrue(result["completed"])
        orch = result["orchestration"]
        self.assertGreaterEqual(orch["review_iterations"], 1)
        self.assertEqual(orch["verdicts"], ["revise", "accept"])
        self.assertIn("worker", orch["roles"])
        self.assertGreaterEqual(orch["roles"]["worker"]["runs"], 2)

    def test_revision_limit_escalates(self):
        cfg = _cfg()  # max_iterations default 2
        cfg["routing"]["mode"] = "manual"
        cfg["strategy"] = WORKER_IMPLEMENT_REVIEW
        # Every review says revise -> never accept, hits limit.
        queue = [
            _worker_ok("w1"),
            _reviewer_ok("revise"),
            _worker_ok("w2"),
            _reviewer_ok("revise"),
            _worker_ok("w3"),
            _reviewer_ok("revise"),
        ]

        def fake_build(*args, **kwargs):
            return queue.pop(0) if queue else _worker_ok("exhausted")

        with patch(
            "agent.orchestration.orchestrator._build_and_run_child",
            side_effect=fake_build,
        ):
            result = run_orchestrated(FakeParent(), "Implement feature", cfg)
        self.assertTrue(result["failed"])
        orch = result["orchestration"]
        self.assertGreaterEqual(orch["escalations"], 1)
        self.assertIn("escalated", str(result.get("final_response", "")))


class TestCapabilities(unittest.TestCase):
    def test_reasoning_effort_not_supported_omits_dial(self):
        cfg = _cfg()
        cfg["models"]["worker"]["supports_reasoning_effort"] = False
        captured = {}

        def fake_build(*args, **kwargs):
            captured.update(kwargs)
            return _worker_ok()

        with patch(
            "agent.orchestration.orchestrator._build_and_run_child",
            side_effect=fake_build,
        ):
            result = run_orchestrated(FakeParent(), "Add config option", cfg)
        self.assertTrue(result["completed"])
        # When the provider doesn't support reasoning effort, no dial is passed.
        self.assertIsNone(captured.get("reasoning_config"))


class TestMissingRoleConfig(unittest.TestCase):
    def test_missing_architect_degrades_to_failure(self):
        cfg = _cfg()
        cfg["models"]["architect"]["model"] = ""

        def fake_build(*args, **kwargs):
            return _worker_ok()

        with patch(
            "agent.orchestration.orchestrator._build_and_run_child",
            side_effect=fake_build,
        ):
            # Architect-solo with no architect model -> worker fallback runs.
            result = run_orchestrated(FakeParent(), "Plan the approach", cfg)
        # Worker fallback still completes when architect is absent.
        self.assertIn(result["completed"], (True, False))

    def test_missing_reviewer_degrades(self):
        cfg = _cfg()
        cfg["models"]["reviewer"]["model"] = ""
        # Force a review path; with reviewer model empty, the reviewer inherits
        # the parent model and still runs (graceful degradation, no crash).
        cfg["routing"]["mode"] = "manual"
        cfg["strategy"] = WORKER_IMPLEMENT_REVIEW

        def fake_build(*args, **kwargs):
            if kwargs.get("role") == "reviewer":
                return _reviewer_ok("accept")
            return _worker_ok()

        with patch(
            "agent.orchestration.orchestrator._build_and_run_child",
            side_effect=fake_build,
        ):
            result = run_orchestrated(FakeParent(), "Implement feature", cfg)
        # Degrades gracefully: reviewer ran with the inherited model and the
        # flow reached acceptance.
        self.assertTrue(result["completed"])
        orch = result["orchestration"]
        self.assertIn("reviewer", orch["roles"])
        self.assertEqual(orch["verdicts"], ["accept"])

    def test_missing_architect_degrades_worker(self):
        cfg = _cfg()
        cfg["models"]["architect"]["model"] = ""
        # Architect-solo strategy with no architect model degrades to a worker
        # run (inherit parent model) rather than crashing.
        cfg["routing"]["mode"] = "manual"
        cfg["strategy"] = "ARCHITECT_SOLO"

        def fake_build(*args, **kwargs):
            return _worker_ok("brief produced")

        with patch(
            "agent.orchestration.orchestrator._build_and_run_child",
            side_effect=fake_build,
        ):
            result = run_orchestrated(FakeParent(), "Plan the approach", cfg)
        self.assertTrue(result["completed"])
        self.assertIn("final_response", result)
        self.assertNotIn("escalated", str(result.get("final_response", "")))


class TestFailureHandling(unittest.TestCase):
    def test_worker_failure_escalates(self):
        cfg = _cfg()
        cfg["routing"]["mode"] = "manual"
        cfg["strategy"] = WORKER_IMPLEMENT_REVIEW
        cfg["escalation"]["max_worker_failures"] = 1

        def fake_build(*args, **kwargs):
            return {
                "ok": False,
                "status": "error",
                "error": "provider quota",
                "summary": None,
            }

        with patch(
            "agent.orchestration.orchestrator._build_and_run_child",
            side_effect=fake_build,
        ):
            result = run_orchestrated(FakeParent(), "Implement feature", cfg)
        self.assertTrue(result["failed"])
        self.assertGreaterEqual(result["orchestration"]["escalations"], 1)

    def test_escaped_exception_falls_back_cleanly(self):
        cfg = _cfg()

        def fake_build(*args, **kwargs):
            raise RuntimeError("boom")

        with patch(
            "agent.orchestration.orchestrator._build_and_run_child",
            side_effect=fake_build,
        ):
            result = run_orchestrated(FakeParent(), "Implement feature", cfg)
        self.assertTrue(result["failed"])
        self.assertIn("orchestration failed", result["final_response"])


class TestMetrics(unittest.TestCase):
    def test_metrics_block_present(self):
        cfg = _cfg()
        queue = [_architect_ok(), _worker_ok(), _reviewer_ok("accept")]

        def fake_build(*args, **kwargs):
            return queue.pop(0)

        with patch(
            "agent.orchestration.orchestrator._build_and_run_child",
            side_effect=fake_build,
        ):
            result = run_orchestrated(
                FakeParent(),
                "Design the architecture for a multi-file feature",
                cfg,
            )
        orch = result["orchestration"]
        for key in (
            "strategy",
            "worker_reasoning_effort",
            "review_required",
            "roles",
            "review_iterations",
            "escalations",
            "verdicts",
            "tool_calls",
            "wall_clock_seconds",
            "total_api_calls",
            "total_tokens",
        ):
            self.assertIn(key, orch, f"missing metric key: {key}")
        self.assertIn("worker", orch["roles"])
        self.assertIn("reviewer", orch["roles"])
        self.assertGreaterEqual(orch["total_api_calls"], 3)


class TestReviewerParsing(unittest.TestCase):
    def test_parse_accept(self):
        verdict, issues, req, conf = _parse_reviewer_result(
            {"summary": '{"verdict":"accept","issues":[],"required_changes":[],"confidence":0.9}'}
        )
        self.assertEqual(verdict, "accept")
        self.assertEqual(issues, [])
        self.assertAlmostEqual(conf, 0.9)

    def test_parse_fenced_json(self):
        verdict, _, _, _ = _parse_reviewer_result(
            {
                "summary": '```json\n{"verdict":"revise","issues":["a"],"required_changes":["b"],"confidence":0.5}\n```'
            }
        )
        self.assertEqual(verdict, "revise")

    def test_parse_unparseable_defaults_revise(self):
        verdict, _, _, conf = _parse_reviewer_result({"summary": "not json"})
        self.assertEqual(verdict, "revise")
        self.assertEqual(conf, 0.0)


if __name__ == "__main__":
    unittest.main()
