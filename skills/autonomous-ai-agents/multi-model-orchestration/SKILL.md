---
name: multi-model-orchestration
description: "Use when a task should be routed across worker/architect/reviewer models. Behavioral guidance + prompt templates for the thin-Python orchestration engine (orchestration.engine: skill); the deterministic loop and metrics live in agent/orchestration/skill_engine.py."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [orchestration, multi-model, routing, architect, reviewer, worker]
    related_skills: [hermes-agent, subagent-driven-development]
---

# Multi-Model Orchestration

## Overview

This skill supplies the **behavioral guidance and prompt templates** for the
thin-Python orchestration engine (Solution 2, selected via
`orchestration.engine: skill` in `config.yaml`). The Python in
`agent/orchestration/skill_engine.py` only enforces the deterministic skeleton —
routing decision, bounded worker↔reviewer loop, escalation depth, and metrics
collection. Everything about *how* each role should behave lives here, in
editable prose.

Why two engines? The original Sol-Advisor was a markdown file instructing the
agent. The all-Python engine (Solution 1, `orchestration.engine: python`) gives
deterministic control flow and reliable metrics but bakes the guidance into
code. This skill-based engine keeps the deterministic loop in code while moving
the flexible, prose-editable guidance into a skill — a middle path to benchmark
head-to-head.

## When to Use

- `orchestration.enabled: true` and `orchestration.engine: skill` in config.
- You want to edit the architect/worker/reviewer behavior as prose, without
  touching Python.
- You are benchmarking Solution 1 (all-Python) vs Solution 2 (skill-driven).

Don't use for: single-model sessions (leave `orchestration.enabled: false`),
or when you want the prompts hard-coded (that is Solution 1).

## Roles

| Role | Model (example) | What it does |
|------|-----------------|--------------|
| `worker` | `deepseek-v4-flash` | implements with normal tools, reports changed files + tests |
| `architect` | `glm-5.2` | compact structured brief; escalates on re-plan |
| `reviewer` | `glm-5.2` | independent review; returns strict JSON verdict |

## Guidance the engine injects into every role prompt

The engine reads this file's body (frontmatter stripped) and injects it into the
architect, worker, and reviewer prompts. Keep this section actionable and
concise — it is literally the shared behavioral contract each role reads.

### For the architect

- Produce a **compact structured implementation brief** only. Do not write a
  long essay.
- Use this exact structure (each section one to a few lines):
  `objective`, `constraints`, `files_components_likely_affected`,
  `implementation_approach`, `acceptance_criteria`, `risks`, `tests_required`.
- Preserve the original user request as authoritative context.
- Do not dump hidden chain-of-thought.

### For the worker

- Complete the task using your normal tools and repository access.
- The original request is authoritative; an architect brief (if any) and
  explicit acceptance criteria refine it.
- If the reviewer sent revision feedback, address it directly.
- When finished, report in your summary: changed files, tests run and their
  results, and any remaining uncertainty.

### For the reviewer

- Perform an **independent** review. Do not continue the worker's chain of
  thought.
- You are given: the original request, acceptance criteria, the worker's
  summary, the worker model, and a tool-call trace. Inspect the repository
  yourself to verify correctness, requirements coverage, tests, regressions,
  safety, and maintainability.
- Respond with **ONLY** a JSON object in this exact shape (no markdown fence,
  no extra text):

```json
{"verdict": "accept" | "revise" | "escalate",
 "issues": [],
 "required_changes": [],
 "confidence": 0.0}
```

- `accept` = meets acceptance criteria, no blocking issues.
- `revise` = fixable issues; list them in `required_changes`.
- `escalate` = task grew / architecture unclear / repeated failures.
- `confidence` = 0.0–1.0 in this verdict.
- Never expose hidden chain-of-thought.

## How the engine uses this file

```
user prompt
   │
   ▼
router.py (deterministic heuristics) ──► RoutingDecision
   │
   ▼
skill_engine.run_skill_orchestrated()
   │  loads this SKILL.md body via _load_skill_guidance()
   │  injects it into architect/worker/reviewer prompts
   │  runs the bounded worker↔reviewer loop (review.max_iterations)
   │  escalates to architect (depth-bounded) on failure/rejection
   │  collects the metrics block
   ▼
result dict: {final_response, api_calls, completed, failed, orchestration:{...}}
```

## Editing this skill

- Edit the "Guidance" section above to change role behavior — no Python changes.
- The prompt templates that assemble this guidance into role prompts live in
  `agent/orchestration/skill_engine.py` (`_architect_prompt`, `_worker_prompt`,
  `_reviewer_prompt`).
- The engine falls back to compact built-in templates if this file cannot be
  found, so a missing/renamed skill degrades gracefully (never breaks the turn).

## Common Pitfalls

1. **Expecting the current session to see a newly added skill.** The skill
   loader is cached at session start; a fresh session is needed for
   `skill_view` to resolve it. The engine itself reads SKILL.md from disk on
   every run, so it picks up edits immediately even mid-session.
2. **Editing prompts and expecting metrics to change.** This skill only affects
   *behavioral guidance*. Routing decisions, revision bounds, and metrics are
   enforced by Python in `skill_engine.py` / `router.py`.
3. **Forgetting `orchestration.enabled: true`.** The engine (and Solution 1) is
   a complete no-op when disabled.

## Verification Checklist

- [ ] `orchestration.enabled: true` and `orchestration.engine: skill` in config.
- [ ] A fresh session; first top-level task turn routes through the orchestrator.
- [ ] Review verdicts returned as strict JSON (check `orchestration.verdicts` in
      the result).
- [ ] Per-role metrics present (`orchestration.roles.*`).
- [ ] `orchestration.enabled: false` behaves exactly as before (no-op).

## One-Shot Recipe

Enable the skill engine and run one task:

```bash
hermes chat -q "Design the architecture for a multi-file feature and implement it" -Q
```

With `config.yaml`:
```yaml
orchestration:
  enabled: true
  engine: skill
  routing: { mode: heuristic }
  models:
    worker:   { model: deepseek-v4-flash, provider: deepseek, reasoning_effort: high }
    architect:{ model: glm-5.2, provider: zai }
    reviewer: { model: glm-5.2, provider: zai }
```
