#!/usr/bin/env python3
"""
Tests for the skill-driven orchestration engine (Solution 2).

Verifies: engine selection in maybe_route_task, skill-load + built-in fallback,
the bounded worker<->reviewer loop, escalation, capability gating, and metrics.
All role children are mocked — no real LLM calls.
"""

import json
import unittest
from unittest.mock import patch

from agent.orchestration.config import load_orchestration_config
from agent.orchestration.skill_engine import (
    _extract_json_object,
    _load_skill_guidance,
    _parse_verdict,
    _strip_frontmatter,
    run_skill_orchestrated,
)


def _cfg(**overrides):
    base = {
        "enabled": True,
        "engine": "skill",
        "routing": {"mode": "manual"},
        "strategy": "WORKER_IMPLEMENT_REVIEW",
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
        "strategy": "WORKER_IMPLEMENT_REVIEW",
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


def _reviewer_ok(verdict="accept"):
    return _worker_ok(
        summary=json.dumps(
            {
                "verdict": verdict,
                "issues": ["x"] if verdict == "revise" else [],
                "required_changes": ["fix x"] if verdict == "revise" else [],
                "confidence": 0.9,
            }
        ),
        model="glm-5.2",
        role="reviewer",
    )


class FakeParent:
    platform = "cli"
    model = "deepseek-v4-flash"
    _session_messages = []


class TestSkillLoad(unittest.TestCase):
    def test_strip_frontmatter(self):
        content = '---\nname: x\ndescription: y\n---\n# Body\nmore'
        self.assertEqual(_strip_frontmatter(content), "# Body\nmore")

    def test_load_skill_guidance_from_repo(self):
        # The in-repo skill lives at skills/autonomous-ai-agents/... — the engine
        # should find it via the repo root (parent.parent.parent of skill_engine.py).
        guidance = _load_skill_guidance()
        self.assertIn("markdown", guidance)
        self.assertIn("architect", guidance["markdown"])
        self.assertIn("reviewer", guidance["markdown"])

    def test_load_skill_guidance_empty_when_not_found(self):
        with patch(
            "agent.orchestration.skill_engine.Path.rglob",
            side_effect=lambda _: iter([]),
        ):
            guidance = _load_skill_guidance()
        self.assertEqual(guidance, {})


class TestSkillEngineFlows(unittest.TestCase):
    def test_worker_implement_review_accept(self):
        cfg = _cfg()
        queue = [_worker_ok(), _reviewer_ok("accept")]

        def fake(*args, **kwargs):
            return queue.pop(0)

        with patch(
            "agent.orchestration.skill_engine._build_and_run_child",
            side_effect=fake,
        ):
            result = run_skill_orchestrated(FakeParent(), "Implement feature", cfg)
        self.assertTrue(result["completed"])
        orch = result["orchestration"]
        self.assertEqual(orch["engine"], "skill")
        self.assertIn("worker", orch["roles"])
        self.assertIn("reviewer", orch["roles"])
        self.assertEqual(orch["verdicts"], ["accept"])

    def test_revision_then_accept(self):
        cfg = _cfg()
        queue = [
            _worker_ok("first"),
            _reviewer_ok("revise"),
            _worker_ok("second"),
            _reviewer_ok("accept"),
        ]

        def fake(*args, **kwargs):
            return queue.pop(0)

        with patch(
            "agent.orchestration.skill_engine._build_and_run_child",
            side_effect=fake,
        ):
            result = run_skill_orchestrated(FakeParent(), "Implement feature", cfg)
        self.assertTrue(result["completed"])
        orch = result["orchestration"]
        self.assertGreaterEqual(orch["review_iterations"], 1)
        self.assertEqual(orch["verdicts"], ["revise", "accept"])

    def test_revision_limit_escalates(self):
        cfg = _cfg()
        queue = [
            _worker_ok("w1"),
            _reviewer_ok("revise"),
            _worker_ok("w2"),
            _reviewer_ok("revise"),
            _worker_ok("w3"),
            _reviewer_ok("revise"),
        ]

        def fake(*args, **kwargs):
            return queue.pop(0) if queue else _worker_ok("exhausted")

        with patch(
            "agent.orchestration.skill_engine._build_and_run_child",
            side_effect=fake,
        ):
            result = run_skill_orchestrated(FakeParent(), "Implement feature", cfg)
        self.assertTrue(result["failed"])
        self.assertGreaterEqual(result["orchestration"]["escalations"], 1)

    def test_reasoning_effort_capability_gating(self):
        cfg = _cfg()
        cfg["models"]["worker"]["supports_reasoning_effort"] = False
        captured = {}

        def fake(*args, **kwargs):
            captured.update(kwargs)
            if kwargs.get("role") == "reviewer":
                return _reviewer_ok("accept")
            return _worker_ok()

        with patch(
            "agent.orchestration.skill_engine._build_and_run_child",
            side_effect=fake,
        ):
            result = run_skill_orchestrated(FakeParent(), "Implement feature", cfg)
        self.assertTrue(result["completed"])
        self.assertIsNone(captured.get("reasoning_config"))

    def test_escaped_exception_clean_failure(self):
        cfg = _cfg()

        def fake(*args, **kwargs):
            raise RuntimeError("boom")

        with patch(
            "agent.orchestration.skill_engine._build_and_run_child",
            side_effect=fake,
        ):
            result = run_skill_orchestrated(FakeParent(), "Implement feature", cfg)
        self.assertTrue(result["failed"])
        self.assertIn("orchestration failed", result["final_response"])
        self.assertIn("orchestration", result)

    def test_low_effort_worker_gets_restricted_toolset(self):
        from agent.orchestration.skill_engine import _role_toolsets

        self.assertEqual(_role_toolsets("worker", "low"), ["file"])
        self.assertEqual(_role_toolsets("worker", "minimal"), ["file"])
        self.assertEqual(_role_toolsets("worker", "high"), ["file", "terminal"])
        self.assertEqual(_role_toolsets("worker", "max"), ["file", "terminal"])
        self.assertEqual(_role_toolsets("architect", "high"), ["file", "terminal"])
        self.assertEqual(_role_toolsets("reviewer", "high"), ["file", "terminal"])


class TestVerdictParsing(unittest.TestCase):
    def test_parse_accept(self):
        verdict, issues, req, conf = _parse_verdict(
            {"summary": '{"verdict":"accept","issues":[],"required_changes":[],"confidence":0.9}'}
        )
        self.assertEqual(verdict, "accept")
        self.assertAlmostEqual(conf, 0.9)

    def test_parse_fenced_json(self):
        verdict, _, _, _ = _parse_verdict(
            {
                "summary": '```json\n{"verdict":"revise","issues":["a"],"required_changes":["b"],"confidence":0.5}\n```'
            }
        )
        self.assertEqual(verdict, "revise")

    def test_parse_unparseable_defaults_revise(self):
        verdict, _, _, conf = _parse_verdict({"summary": "not json"})
        self.assertEqual(verdict, "revise")
        self.assertEqual(conf, 0.0)


class TestEngineSelection(unittest.TestCase):
    def test_engine_selector_routes_to_skill(self):
        from agent.orchestration import maybe_route_task

        cfg = _cfg()
        with patch(
            "agent.orchestration.config.load_orchestration_config",
            return_value=cfg,
        ):
            # patch the skill engine's child seam to avoid real LLM calls
            queue = [_worker_ok(), _reviewer_ok("accept")]

            def fake(*args, **kwargs):
                return queue.pop(0)

            with patch(
                "agent.orchestration.skill_engine._build_and_run_child",
                side_effect=fake,
            ):
                result = maybe_route_task(FakeParent(), "Implement feature")
        self.assertNotIn("orchestration_disabled", result)
        self.assertEqual(result["orchestration"]["engine"], "skill")

    def test_engine_disabled_returns_marker(self):
        from agent.orchestration import maybe_route_task

        with patch(
            "agent.orchestration.config.load_orchestration_config",
            return_value=load_orchestration_config({"orchestration": {"enabled": False}}),
        ):
            result = maybe_route_task(FakeParent(), "anything")
        self.assertTrue(result.get("orchestration_disabled"))


if __name__ == "__main__":
    unittest.main()
