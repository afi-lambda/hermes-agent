# Multi-Model Orchestration (Sol-Advisor-style)

> **Status:** experimental. Disabled by default. When disabled, Hermes behaves
> **exactly** as before — this feature is a fully opt-in layer on top of the
> normal single-model agent loop.

This document explains how to use the orchestration layer, how to activate it,
how to keep using Hermes without it, how it works internally, how it was
implemented, how the tests work, and how to run a LiveCodeBench benchmark
against it.

---

## 1. What it is

A config-driven layer that routes a task to generic **roles** instead of a
single model:

| Role | Purpose | Default model (example) |
|------|---------|--------------------------|
| `worker` | implements the task with normal tools | `deepseek-v4-flash` (cheap/fast) |
| `architect` | produces a compact implementation brief / escalates | `glm-5.2` (strong) |
| `reviewer` | independent review of the worker's result | `glm-5.2` (strong) |

The role→model mapping is **configuration, not code**. The layer is
model-independent: you can point `worker`/`architect`/`reviewer` at any
provider:model pair Hermes supports.

The layer has **two interchangeable engines** (select via
`orchestration.engine`), so you can benchmark them head-to-head:

| Engine | `orchestration.engine` | Design |
|--------|------------------------|--------|
| **Solution 1 — all-Python** | `python` (default) | routing + loop + **prompts** all in Python (`orchestrator.py`, `prompts.py`) |
| **Solution 2 — skill-driven** | `skill` | thin Python core (loop + metrics) + **behavioral guidance/prompts** in a skill file (`skill_engine.py` + `skills/autonomous-ai-agents/multi-model-orchestration/SKILL.md`) |

The central experiment it enables:

> **Can Hermes reach near-strong-model reliability while shifting most
> implementation tokens and tool work to a cheap model (DeepSeek-V4-Flash)?**

---

## 2. How to activate it

Edit `~/.hermes/config.yaml` (or the profile you use) and add:

```yaml
orchestration:
  enabled: true
  engine: python           # python (all-Python) | skill (skill-driven)

  routing:
    mode: heuristic        # heuristic | manual

  models:
    worker:
      model: deepseek-v4-flash
      provider: deepseek
      reasoning_effort: high      # none|minimal|low|medium|high|xhigh|max|ultra
    architect:
      model: glm-5.2
      provider: zai
      reasoning_effort: ""
    reviewer:
      model: glm-5.2
      provider: zai
      reasoning_effort: ""

  review:
    enabled: true
    max_iterations: 2

  escalation:
    enabled: true
    max_worker_failures: 3

  strategy: ""             # only used when routing.mode: manual
```

Notes:

- **`provider` is optional.** If you leave it empty, the role inherits the
  parent agent's provider/credentials (same contract as `delegation.model`).
  Set it when the role's model lives on a different provider (e.g. DeepSeek vs
  Z.ai/GLM).
- **`reasoning_effort` is optional and capability-gated.** If a provider/model
  does not support `reasoning_effort`, the layer sends **no effort dial at all**
  (it never assumes support). Set `supports_reasoning_effort: false` on a role
  to force this.
- **`routing.mode: manual`** ignores the heuristics and always uses the
  `strategy` you set — this is how you force one of the five benchmark modes.

### Restart

Config is read at agent startup. After editing, start a **fresh** Hermes
session (`/new` in chat, or relaunch `hermes`). The orchestration hook only
fires on a **fresh task turn** of a **top-level** agent.

---

## 3. How to use Hermes WITHOUT it

**Do nothing.** Orchestration is disabled by default (`enabled: false`). With it
disabled:

- `AIAgent.run_conversation` short-circuits the orchestration prelude
  immediately — zero behavior change.
- Model selection, tool calling, delegation, memory, skills, gateway, cron —
  all work exactly as before.
- The only code touched on the normal path is a guarded `try/except` block that
  returns immediately when disabled.

If you have it enabled and want to turn it off, set `orchestration.enabled:
false` (or remove the block) and start a fresh session.

Even when enabled, the layer is **fail-safe**: any error in routing, config, or
a role child degrades to the normal single-model loop rather than breaking the
turn.

### Choosing between the two engines

- **`engine: python`** (default): the architect/worker/reviewer prompts are
  hard-coded in `agent/orchestration/prompts.py`. Deterministic, self-contained,
  no skill dependency.
- **`engine: skill`**: the prompts and behavioral guidance live in the
  `multi-model-orchestration` skill at
  `skills/autonomous-ai-agents/multi-model-orchestration/SKILL.md`. The Python
  (`agent/orchestration/skill_engine.py`) only enforces the loop skeleton and
  metrics, and reads the skill body on every run. Edit the skill's "Guidance"
  section to change role behavior without touching Python. If the skill can't be
  found, the engine falls back to compact built-in templates.

To switch, set `orchestration.engine` and start a fresh session. The metrics
block includes `"engine": "python" | "skill"` so benchmark results are
attributable.

---

## 4. How it works (architecture)

### The single integration point

The only change to the core hot path is a guarded prelude at the top of
`AIAgent.run_conversation` (`run_agent.py`). This method is the choke point
shared by CLI `-q`, interactive first turn, gateway, and oneshot.

```python
# run_agent.py — AIAgent.run_conversation
try:
    from agent.orchestration import maybe_route_task, orchestration_should_run
    if orchestration_should_run(self, conversation_history):
        routed = maybe_route_task(self, str(user_message), conversation_history)
        if routed and not routed.get("orchestration_disabled"):
            return routed
except Exception as exc:
    logger.warning("orchestration prelude failed, falling back to normal loop: %s", exc)
# ... existing body unchanged ...
```

`orchestration_should_run` returns `True` only when **all** of:
1. orchestration is enabled in config,
2. the agent is **not** a subagent (`platform != "subagent"`),
3. this is a **fresh task turn** (no prior turns in `conversation_history` and
   no prior user/assistant turns in `self._session_messages`).

So mid-conversation follow-ups and nested subagents are never re-routed.

### The pipeline

```
user prompt
   │
   ▼
[router]  ── deterministic heuristics ──►  RoutingDecision
   │                                        {strategy, worker_reasoning_effort,
   │                                         review_required, reason}
   ▼
[orchestrator]  runs the chosen strategy
   │
   ├─ WORKER_SOLO                          → worker child
   ├─ ARCHITECT_SOLO                       → architect child (brief)
   ├─ ARCHITECT_PLAN_WORKER_IMPLEMENT_REVIEW
   │     architect brief → worker → reviewer (bounded loop)
   └─ WORKER_IMPLEMENT_REVIEW
         worker → reviewer (bounded loop)
   │
   ▼
result dict shaped like run_conversation's:
   {final_response, api_calls, completed, failed, orchestration:{...metrics...}}
```

### Roles run as sub-agents

Each role is executed as an **isolated child agent** built with Hermes's
existing sub-agent primitive (`tools.delegate_tool._build_child_agent` /
`_run_single_child`). This reuses, rather than reimplements:

- toolset isolation and the child tool blocklist,
- credential / provider resolution (`resolve_runtime_provider`),
- retry / error handling,
- token and cost accounting (`session_prompt_tokens`, `session_estimated_cost_usd`).

The child gets a focused system prompt built from the role's goal (architect
brief, worker prompt, or reviewer prompt) — the full parent conversation is
**not** copied into the child.

### The reviewer

The reviewer is an **independent** review (fresh context, not a continuation of
the worker). It receives:

- the original request,
- acceptance criteria (from the architect brief, if any),
- the worker's summary,
- the worker's tool-call trace,
- observed execution metrics.

It returns strict JSON:

```json
{"verdict": "accept" | "revise" | "escalate",
 "issues": [],
 "required_changes": [],
 "confidence": 0.0}
```

No hidden chain-of-thought is requested or exposed. The review focuses on
observable correctness, requirements, tests, regressions, safety, and
maintainability.

### The revision loop

Bounded by `review.max_iterations` (default 2):

```
worker → reviewer
   ├─ accept  → done
   ├─ revise  → worker revision → reviewer (repeat, up to max_iterations)
   └─ escalate / budget exhausted → escalate to architect or return failure
```

Escalation is depth-bounded (one architect re-plan) so a persistent rejection
cannot recurse forever — it returns a clear failure instead.

### Metrics

The returned result carries an `orchestration` block with structured metrics for
later benchmarking:

```json
{
  "strategy": "ARCHITECT_PLAN_WORKER_IMPLEMENT_REVIEW",
  "worker_reasoning_effort": "max",
  "review_required": true,
  "reason": "architectural/ambiguous/high-risk task: ...",
  "roles": {
    "architect": {"model": "glm-5.2", "api_calls": 1, "tokens": {"input": 10, "output": 5}, "duration_seconds": 1.2, "cost_usd": 0.0, "tool_calls": 0, "runs": 1},
    "worker":    {"model": "deepseek-v4-flash", "api_calls": 3, "tokens": {"input": 100, "output": 50}, "duration_seconds": 2.0, "cost_usd": 0.001, "tool_calls": 1, "runs": 1},
    "reviewer":  {"model": "glm-5.2", "api_calls": 1, "tokens": {"input": 20, "output": 10}, "duration_seconds": 0.8, "cost_usd": 0.0, "tool_calls": 0, "runs": 1}
  },
  "review_iterations": 0,
  "escalations": 0,
  "verdicts": ["accept"],
  "tool_calls": 1,
  "wall_clock_seconds": 4.0,
  "total_api_calls": 5,
  "total_tokens": {"input": 130, "output": 65}
}
```

No persistent analytics store is built — the metrics ride on the result dict so
a benchmark harness can collect them.

---

## 5. How it was implemented

### Files

New (`agent/orchestration/`):

| File | Responsibility |
|------|----------------|
| `config.py` | `DEFAULT_ORCHESTRATION`, `load_orchestration_config()`, per-role model/effort resolution, `build_role_reasoning_config()` capability gate, `engine` selector |
| `router.py` | deterministic heuristic router → `RoutingDecision`; keyword/dimension scoring; manual-mode override |
| `prompts.py` | compact architect-brief / worker / reviewer prompt builders (Solution 1, all-Python) |
| `orchestrator.py` | Solution 1 strategy driver: revision loop, escalation, metrics |
| `skill_engine.py` | Solution 2 thin core: loop skeleton + metrics; loads guidance from the skill file |
| `__init__.py` | `orchestration_should_run()` + `maybe_route_task()` (the hook surface); dispatches to the selected engine |

Modified:

| File | Change |
|------|--------|
| `hermes_cli/config_defaults.py` | added the `orchestration:` default block (disabled by default) |
| `run_agent.py` | guarded prelude at the top of `AIAgent.run_conversation` |

### Design decisions

- **Reuse over rewrite.** Role children are built through `delegate_tool`'s
  existing sub-agent machinery — no parallel agent framework, no new scheduler,
  no DAG engine, no persistent multi-agent memory.
- **Two interchangeable engines** (`orchestration.engine`): the all-Python
  Solution 1 (`orchestrator.py` + `prompts.py`) and the skill-driven Solution 2
  (`skill_engine.py` + the `multi-model-orchestration` skill). Both share the
  same router, role config, child-build seam, and metrics schema, so they are
  benchmark-identical except for where the behavioral guidance lives (code vs.
  editable skill prose). The original Sol-Advisor was markdown-only; these two
  engines let you measure the all-code vs. skill-guided tradeoff.
- **Config is the only place models are named.** The code knows `worker`,
  `architect`, `reviewer`; DeepSeek/GLM live in `config.yaml`.
- **Backward compatibility is structural.** The hook is a no-op when disabled,
  and even when enabled it only fires on a fresh top-level task turn and falls
  back on any error.
- **Reasoning effort is capability-gated.** `build_role_reasoning_config`
  returns `None` (no dial) when the role's provider doesn't support it.
- **Escalation is depth-bounded** to prevent infinite recursion.

### The five benchmark modes map to strategies

| Mode | Strategy |
|------|----------|
| A. GLM-5.2 alone | `ARCHITECT_SOLO` (or `WORKER_SOLO` with worker→glm) |
| B. DeepSeek-V4-Flash max alone | `WORKER_SOLO` with worker effort `max` |
| C. GLM plan → DeepSeek high → GLM review | `ARCHITECT_PLAN_WORKER_IMPLEMENT_REVIEW` |
| D. DeepSeek high → GLM review | `WORKER_IMPLEMENT_REVIEW` |
| E. Adaptive routing | `routing.mode: heuristic` (router picks low/high/max + review) |

---

## 6. How the tests work

Tests live in `tests/agent/orchestration/`. They use **mocks** — no real LLM
calls, no API keys.

### `test_router.py` (16 tests)

Tests the deterministic router against the spec's routing policy:

- tiny/mechanical task → `WORKER_SOLO`, effort `low`
- ordinary coding → `WORKER_SOLO`, effort `high`
- difficult/debug → effort `max`
- architectural/ambiguous/high-risk → `ARCHITECT_PLAN_WORKER_IMPLEMENT_REVIEW`
- manual strategy override (`routing.mode: manual`)
- `RoutingDecision.to_dict()` shape

### `test_orchestrator.py` (19 tests)

Tests the Solution 1 (all-Python) strategy driver by patching the role-child
seam (`agent.orchestration.orchestrator._build_and_run_child`) to return canned
per-role results. Covers:

- orchestration disabled → `orchestration_should_run` returns `False`
- subagent / follow-up turn → not routed
- fresh task turn → routed
- `WORKER_SOLO` flow
- architect→worker→reviewer-accept flow
- reviewer requests revision → worker revision → accept
- revision-limit reached → escalation / failure
- provider without reasoning-effort support → no effort dial passed
- missing architect/reviewer model → graceful degradation (inherit parent)
- worker failure → escalation
- escaped exception → clean failure (never breaks the turn)
- metrics block present and populated
- reviewer JSON verdict parsing (accept / fenced JSON / unparseable→revise)

### `test_skill_engine.py` (13 tests)

Tests the Solution 2 (skill-driven) thin core by patching
`agent.orchestration.skill_engine._build_and_run_child`. Covers: skill loading
(repo + fallback), frontmatter stripping, worker→reviewer accept, revision→
accept, revision-limit escalation, reasoning-effort capability gating, clean
failure on exception, verdict parsing, and `maybe_route_task` engine dispatch.

### How to run them

```bash
cd /home/alain/.hermes/scratch/hermes-orchestration
HERMES_HOME=$(mktemp -d) venv/bin/python -m pytest tests/agent/orchestration/ -o 'addopts=' -q
# 48 passed (16 router + 19 orchestrator + 13 skill_engine)
```

Regression (delegation + config + core loop):

```bash
HERMES_HOME=$(mktemp -d) venv/bin/python -m pytest \
  tests/tools/test_delegate.py \
  tests/hermes_cli/test_config_validation.py \
  tests/hermes_cli/test_config_env_expansion.py \
  tests/run_agent/test_run_agent.py -o 'addopts=' -q
```

> Note: the scratch clone's `venv` is a symlink to the live checkout's venv. If
> a test fails with `ImportError: The 'anthropic' package is required`, install
> it: `venv/bin/python -m pip install 'anthropic>=0.39.0'`.

---

## 7. How to run LiveCodeBench against it

There is a ready-made LiveCodeBench harness at
`/home/alain/benchmarks/livecodebench/`. It is **not** part of the Hermes repo —
it calls a model directly via the OpenAI-compatible `/v1/chat/completions`
endpoint and evaluates generated code against the fixture's test cases.

### The fixture

- `phase1_pilot_50.jsonl` — the deterministic 50-problem pilot (seed 20260812:
  30 AtCoder + 20 LeetCode) used for the Ternary Bonsai 27B pilot.
- `post_cutoff_release_v6_20241001_20250430.jsonl` — a larger post-cutoff
  release (for harder, contamination-free evaluation).

### Run a single model (baseline)

```bash
cd /home/alain/benchmarks/livecodebench
OLLAMA_API_KEY=... MODEL=glm-5.2 THINK_LEVEL=medium python3 run_lcb_ollama_cloud.py
```

Env vars: `MODEL`, `TEMPERATURE` (default 0.2), `MAX_TOKENS` (8192),
`THINK_LEVEL` (`""|low|medium|high|xhigh|max`), `LCB_DATA` (fixture path),
`RPM_LIMIT` (20), `OLLAMA_CLOUD_URL` (default `https://ollama.com/v1`).

Results are written to
`results_<model>_ollama-cloud_think-<level>_lcb_v6_pilot50_<timestamp>.json`
with per-problem pass/fail, token usage, timing, and a summary.

### Run the five orchestration modes

To benchmark the orchestration layer itself, you drive it through Hermes (the
orchestrator builds role children that call the models), then collect the
`orchestration` metrics dict. Two approaches:

**A. Interactive / one-shot through Hermes.** Enable orchestration in config
(section 2), then run a task:

```bash
hermes chat -q "Solve this LiveCodeBench problem: <problem>" -Q
```

The returned result's `orchestration` block gives you strategy, per-role
tokens/cost, review iterations, and verdicts. For a full 50-problem sweep you
would loop over the fixture and feed each problem as a one-shot prompt, then
aggregate the `orchestration` blocks.

**B. Direct harness (no Hermes).** The existing `run_lcb_ollama_cloud.py`
already measures single-model pass@1. To compare orchestration modes, run each
mode's model/effort combination directly:

```bash
# Mode B: DeepSeek-V4-Flash max alone
MODEL=deepseek-v4-flash THINK_LEVEL=max python3 run_lcb_ollama_cloud.py

# Mode A: GLM-5.2 alone
MODEL=glm-5.2 THINK_LEVEL=medium python3 run_lcb_ollama_cloud.py
```

Then compare pass@1, tokens, and cost across the result JSONs. The orchestration
layer's value (modes C/D/E) is measured by the **review iterations and
escalations** it saves vs. the raw single-model pass rate — collect the
`orchestration` metrics from approach A for that.

### Recommended comparison

| Mode | Command / config |
|------|------------------|
| A. GLM-5.2 alone | `MODEL=glm-5.2` direct, or `WORKER_SOLO` with worker→glm |
| B. DeepSeek max alone | `MODEL=deepseek-v4-flash THINK_LEVEL=max` direct, or `WORKER_SOLO` effort `max` |
| C. GLM plan → DeepSeek high → GLM review | `ARCHITECT_PLAN_WORKER_IMPLEMENT_REVIEW` |
| D. DeepSeek high → GLM review | `WORKER_IMPLEMENT_REVIEW` |
| E. Adaptive | `routing.mode: heuristic` |

---

## 8. Known limitations

- The router is keyword/dimension heuristics only — no ML, no model-assisted
  routing yet (documented future hook).
- The reviewer gets the worker summary + tool trace, not a full diff; it
  inspects the repository itself.
- No persistent analytics store — metrics ride on the returned result dict.
- Interactive streaming callbacks are not wired through the orchestration
  return (fine for benchmarking; noted for future work).
- Orchestration only fires on a **fresh top-level task turn**; it does not
  re-route mid-conversation follow-ups.

## 9. Recommended next experiment

Run the five modes (A–E) over the 50-problem pilot fixture, collect the
`orchestration` metrics dict per mode, and compare success rate vs.
tokens/cost/review-iterations. The central question: **can Hermes reach
near-strong-model reliability while shifting most implementation tokens and tool
work to DeepSeek-V4-Flash?**
