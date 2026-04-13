---
title: Tool Orchestrator Environment
emoji: 🔧
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - tool-use
  - cost-aware
---

# ToolOrchestratorEnv

**An OpenEnv-compatible reinforcement learning environment for multi-tool, cost-aware question answering.**

Built on top of [SearchEconomicsEnv](https://huggingface.co/spaces/yashu2000/search-economics-env) (Yash Sharma, USC / Ceramic AI), this environment generalises the single-tool (search-only) formulation to a full **tool-selection problem**: the agent must choose *which* of six tools to call at each step, managing a shared cost budget across a multi-domain question set (HotpotQA, MATH, GPQA, HumanEval).

The core research question: **can an RL agent learn a cost-aware tool routing policy that outperforms simple heuristics like "always search" or "always use the cheapest tool"?**

---

## What the agent learns

Each episode the agent receives 10 questions sampled across four domains. At every step it sees:

- The current **question** and its **domain** tag
- Its **remaining budget** (shared across all questions)
- The **context window** — concatenated outputs from prior tool calls on this question

It picks one action from six tools:

| Tool | `tool_id` | Cost | Best for |
|---|---|---|---|
| Ceramic web search | `ceramic_search` | 1.0 | Multi-hop factual QA |
| Wikipedia lookup | `wiki_lookup` | 0.5 | Entity facts, definitions |
| Calculator | `calculator` | 0.1 | Arithmetic, symbolic math |
| Python executor | `code_executor` | 0.3 | HumanEval code tasks |
| LLM reasoning | `llm_reason` | 2.0 | Graduate-level GPQA problems |
| Commit answer | `commit` | 0.0 | Submit and move to next question |

**The RL objective:** maximise accuracy across all questions while staying within the total budget — learning *which tool to call*, in *which order*, and *when to stop and commit*.

---

## Reward formula

```
On tool call:   R = -tool_cost

On commit:      R = base + η · γ · budget_remaining_ratio

  base     = incorrect_reward + quality · (correct_reward − incorrect_reward)
  quality  = max(ExactMatch, TokenF1)
  η        = 1  if quality ≥ efficiency_bonus_threshold, else 0
  γ        = efficiency_bonus_weight
```

The efficiency bonus is only awarded when the agent answers correctly **and** still has budget remaining — directly incentivising both accuracy and frugality.

---

## Quickstart (local)

```bash
# 1. Clone and install
git clone <this-repo>
cd claude_toolOrchestrator
pip install -r requirements.txt

# 2. Configure keys (copy the example and fill in values)
cp .env.example .env
# Set CERAMIC_API_KEY — sign up free at https://ceramic.ai

# 3. Start the server
uvicorn app:app --port 8000

# 4. Try the interactive demo UI
open http://localhost:8000/web
# or browse the full OpenAPI spec at
open http://localhost:8000/docs
```

---

## HTTP API

### `POST /reset`

Start a new episode. Returns `session_id`, initial `observation`, and `state`.

```json
{ "seed": 42, "config_overrides": { "total_budget": 30.0, "num_questions": 5 } }
```

### `POST /step?session_id=<id>`

Execute one tool call. Pass `session_id` (from `/reset`) as a query param to support parallel agents.

```json
{ "tool_id": "ceramic_search", "query": "When was the Eiffel Tower built?" }
{ "tool_id": "calculator",     "expression": "sqrt(144) + 3" }
{ "tool_id": "code_executor",  "code_snippet": "print(2 ** 10)" }
{ "tool_id": "commit",         "answer": "1889" }
```

### `GET /health`

Returns `{"status": "ok"}`.

---

## Project layout

```
claude_toolOrchestrator/
│
├── app.py                  # FastAPI server — multi-session, OpenAPI, demo UI
├── openenv.yaml            # OpenEnv deployment spec
├── requirements.txt        # Python dependencies
├── .env.example            # Key template (copy → .env, never commit .env)
│
├── env/                    # ── Core RL environment ──────────────────────────
│   ├── environment.py      # ToolOrchestratorEnvironment: reset() + step()
│   ├── models.py           # Pydantic types: Action, Observation, State, ToolResult
│   ├── config.py           # EnvConfig dataclass: budget, costs, reward weights
│   ├── answer_grading.py   # grade() → (exact_match, f1, quality)
│   └── reward.py           # step_reward() + commit_reward()
│
├── ceramic/                # ── Retrieval backend ────────────────────────────
│   └── client.py           # CeramicClient (live) + FallbackCeramicClient (offline)
│
├── data/                   # ── Dataset loading ──────────────────────────────
│   └── loader.py           # load_all() → flat list from 4 HF datasets
│
├── tools/                  # ── Six tool implementations ─────────────────────
│   ├── ceramic_search.py   # Web search (Ceramic AI API)
│   ├── wiki_lookup.py      # Wikipedia REST API, first paragraph
│   ├── calculator.py       # Safe AST-based math evaluator (no exec)
│   ├── code_executor.py    # Sandboxed Python exec (blocks os/sys/subprocess)
│   ├── llm_reason.py       # Together AI chain-of-thought (graceful fallback)
│   └── commit.py           # Answer pass-through; grading runs in environment
│
└── baselines/              # ── Reference policies ───────────────────────────
    ├── random_tool.py      # Uniform random tool selection
    ├── cheapest_first.py   # Always picks cheapest non-commit tool first
    └── oracle.py           # Domain-aware heuristic (search for QA, calc for math)
```

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `CERAMIC_API_KEY` | Yes (for live search) | Ceramic AI key — `POST /search` endpoint |
| `SEE_CERAMIC_API_KEY` | Alternative | HF Spaces alias used by SearchEconomicsEnv |
| `TOGETHER_API_KEY` | Optional | Enables the `llm_reason` tool via Together AI |
| `HF_TOKEN` | Optional | Required only to load gated datasets (GPQA) |

If no Ceramic key is set, `ceramic_search` falls back to deterministic offline results; all other tools work without any key.

---

## Running baselines

```bash
# From inside claude_toolOrchestrator/
python -m baselines.random_tool
python -m baselines.cheapest_first
python -m baselines.oracle
```

---

## Relation to SearchEconomicsEnv

| | [SearchEconomicsEnv](https://github.com/sharma-yash01/SearchEconomicsEnv) | ToolOrchestratorEnv |
|---|---|---|
| Tools available | 1 (search only) | 6 (search, wiki, calc, code, LLM, commit) |
| Datasets | HotpotQA | HotpotQA + MATH + GPQA + HumanEval |
| Budget unit | # of search calls | cost units per tool (tool-specific) |
| Reward shape | Weitzman search penalty | Same formula, extended to tool costs |
| Core RL challenge | *How many* searches to do | *Which* tool to call, in which order |
| Retrieval backend | Ceramic AI | Ceramic AI (shared) |

---

## Docker (HuggingFace Spaces)

```bash
docker build -t tool-orchestrator-env:latest .
docker run -p 8000:8000 -e CERAMIC_API_KEY=cer_sk_live_... tool-orchestrator-env:latest
```

---

## Datasets

- **HotpotQA** — Yang et al., 2018. Multi-hop reasoning over Wikipedia.
- **MATH** — Hendrycks et al., 2021. Competition math levels 3–5.
- **GPQA** — Rein et al., 2023. Graduate-level science QA.
- **HumanEval** — Chen et al., 2021. Python programming tasks.

---

## About

ToolOrchestratorEnv extends SearchEconomicsEnv to a multi-tool setting, framing cost-aware tool selection as the core RL objective. Built for the OpenEnv competition track at AgentX (Berkeley RDI). Ceramic AI search API powers live web retrieval.
