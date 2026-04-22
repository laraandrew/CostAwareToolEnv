# ToolOrchestratorEnv — Research Document

**Authors:** Andrew Lara (Franklin and Marshall College); Yashaswi Sharma, Defu Cao, Muyan Weng (University of Southern California)
**Built on:** [SearchEconomicsEnv](https://github.com/sharma-yash01/SearchEconomicsEnv)
**Live environment:** https://huggingface.co/spaces/landrew9/ToolOrchestratorEnv

**Submission blog:** https://huggingface.co/spaces/landrew9/ToolOrchestratorEnv-Blog

**GitHub:** https://github.com/laraandrew/ToolOrchestratorEnv

---

## Table of Contents

1. [The Problem We're Solving](#1-the-problem-were-solving)
2. [Why Reinforcement Learning?](#2-why-reinforcement-learning)
3. [The Environment: How It Works](#3-the-environment-how-it-works)
4. [The Six Tools](#4-the-six-tools)
5. [The Reward Formula — Deep Dive](#5-the-reward-formula--deep-dive)
6. [The Datasets](#6-the-datasets)
7. [Answer Grading](#7-answer-grading)
8. [The Baselines](#8-the-baselines)
9. [Where This Came From: SearchEconomicsEnv](#9-where-this-came-from-searcheconomicsenv)
10. [What a Trained Agent Should Learn](#10-what-a-trained-agent-should-learn)
11. [File-by-File Reference](#11-file-by-file-reference)

---

## 1. The Problem We're Solving

Modern AI agents have access to tools — search engines, calculators, code runners, databases. Every real-world deployment charges for these tools in some way: API fees, latency, rate limits, compute time. But almost every existing RL benchmark treats tools as **free and unlimited**.

This creates a gap between research and reality. An agent trained on "use whatever tools you want" will behave terribly in production where every call costs money.

**ToolOrchestratorEnv closes that gap.**

The agent is given a fixed **budget** (default: 50 cost units) to spend across 10 questions. Every tool call deducts from that budget. The agent must decide:

- *Which tool is worth calling for this question?*
- *How many times should I call tools before just committing an answer?*
- *Is it worth spending 2.0 on an LLM call, or can a 0.1 calculator solve this?*

This is the same tradeoff a human researcher faces every day.

### Current Research Contribution

This submission contributes a complete, deployed OpenEnv-compatible environment rather than a claimed converged policy. The completed work includes:

- A FastAPI/OpenEnv environment with reset, step, health, tool manifest, browser demo, and concurrent session support.
- Six implemented tools with explicit heterogeneous costs: Ceramic search, Wikipedia lookup, calculator, Python executor, LLM reasoning, and commit.
- A deterministic reward implementation with step costs, Exact Match / token F1 answer grading, a quality-gated efficiency bonus, and a shared 10-question episode budget.
- Three reference baselines: random tool selection, cheapest-first routing, and a domain-aware oracle.
- Unit tests covering API behavior, tool behavior, sandbox restrictions, and core integration paths.

We do **not** claim a converged GRPO checkpoint in the current submission. The environment is built so that GRPO training can now test whether learned policies beat the domain oracle on cost-adjusted reward.

We also do **not** include training logs from Env Factory yet. The submission environment requires repeated structured multi-tool calls across each episode, and we were unable to make that multi-tool action flow reliable inside the current Env Factory integration path before the deadline. The planned next step is to continue experimentation through post-training once Env Factory stabilizes for this interaction pattern and more compatible model series are available.

---

## 2. Why Reinforcement Learning?

RL is the right framework here for three reasons:

**Delayed rewards.** You don't know if a tool call was helpful until you commit your answer at the end of a question. The agent must learn to assign credit backwards — "that search 3 steps ago is why I got this right."

**Exploration.** The agent must try different tool combinations to discover which work best per domain. Supervised learning can't teach this because there's no labeled "correct tool sequence" — there are many valid strategies.

**Multi-step planning.** Each episode has 10 questions and a shared budget. A good agent doesn't just optimize one question — it plans across the whole episode, knowing that spending too much early leaves nothing for later.

---

## 3. The Environment: How It Works

An **episode** works like this:

```
START EPISODE
  Budget = 50.0 units
  Draw 10 questions (mix of domains)

  FOR each question:
    Show agent: question text, domain, remaining budget, context window

    LOOP:
      Agent picks a tool and sends a query
      Environment runs the tool, charges the cost, returns results
      Results are added to the agent's context window for this question

      IF agent calls "commit":
        Grade the answer (Exact Match + Token F1)
        Calculate reward
        Clear context window
        Move to next question
        BREAK

      IF steps_on_this_question >= 8:
        Auto-advance (no commit reward bonus)
        BREAK

      IF budget_spent >= total_budget:
        Episode ends immediately

END EPISODE
```

The agent sees an **observation** at every step containing:
- The question text and domain tag
- How much budget is left and what fraction of the total that is
- What tools it already called on this question and what they returned
- How many questions remain
- Running accuracy so far

---

## 4. The Six Tools

Each tool is a Python function that takes the agent's action and returns a result. Here's exactly what each one does, technically:

---

### `ceramic_search` — Cost: **1.0**

**What it does:** Sends a POST request to the [Ceramic AI](https://ceramic.ai) search API (`https://api.ceramic.ai/search`) with the agent's query. Returns up to 5 web results with title, URL, and a description snippet.

**Best for:** HotpotQA questions that require finding factual information spread across multiple web sources.

**Technical note:** No pagination parameter is supported — Ceramic returns up to 10 results per call, we slice to the top 5. The API requires only `{"query": "..."}` in the POST body.

**Fallback:** If no `CERAMIC_API_KEY` is set, `FallbackCeramicClient` generates deterministic fake results using SHA-256 hashing — so tests are reproducible offline.

```python
# What the tool receives:
action.query = "Who was the first person to walk on the moon?"

# What it sends to Ceramic:
POST https://api.ceramic.ai/search
{"query": "Who was the first person to walk on the moon?"}

# What the agent gets back (in context_window):
"[ceramic_search] **NASA Moon Landing** (score: 892.3)
Neil Armstrong became the first human to step onto the Moon...

**Apollo 11 Mission** (score: 741.1)
On July 20, 1969, Neil Armstrong and Buzz Aldrin landed..."
```

---

### `wiki_lookup` — Cost: **0.5**

**What it does:** Hits the Wikipedia REST API (`https://en.wikipedia.org/api/rest_v1/page/summary/{title}`) and returns the first paragraph of the article for the queried topic. No API key required — Wikipedia is free.

**Best for:** Factual entity lookups where you know the subject (e.g., "Albert Einstein", "World War II"). Cheaper than search and more reliable for well-known topics.

**Technical note:** The query string becomes the Wikipedia article title (spaces replaced with underscores). Returns a 404 error result if the article doesn't exist — the agent can then try a different query.

```python
action.query = "William Shakespeare"
# Hits: https://en.wikipedia.org/api/rest_v1/page/summary/William_Shakespeare
# Returns: "William Shakespeare was an English playwright, poet and actor..."
```

---

### `calculator` — Cost: **0.1**

**What it does:** Evaluates a math expression safely using Python's `ast` (Abstract Syntax Tree) module. The expression is parsed into a tree structure, and only pre-approved operations are allowed (addition, subtraction, multiplication, division, power, modulo, comparisons, and common math functions like `sqrt`, `log`, `sin`, `cos`).

**Why not just use `eval()`?** Because `eval("__import__('os').system('rm -rf /')")` would delete your hard drive. The AST approach means the code is never executed — it's parsed into a data structure and we only compute what we explicitly allow.

**Best for:** MATH competition problems, any arithmetic, symbolic computations.

```python
action.expression = "sqrt(144) + 3 * 7"
# Returns: "23.0"

action.expression = "2 ** 10"
# Returns: "1024"

action.expression = "import os"  # BLOCKED — not a valid math expression
# Returns: "[Calc error: Unsupported AST node: Import]"
```

---

### `code_executor` — Cost: **0.3**

**What it does:** Runs Python code in a sandboxed `exec()` environment for intended coding tasks. Captures whatever is printed to stdout and returns it as the result.

**Security model:** Blocks import statements, dangerous builtin names such as `open`, `eval`, `exec`, `globals`, and obvious object-graph escape paths such as dunder attribute traversal. Only a curated builtin/module surface is exposed.

**Best for:** HumanEval coding tasks where the agent needs to actually run code to verify correctness.

```python
action.code_snippet = """
def fibonacci(n):
    if n <= 1: return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(10))
"""
# Returns: "55"
```

---

### `llm_reason` — Cost: **2.0**

**What it does:** Sends the query to a large language model via the [Together AI](https://together.ai) API (default model: `meta-llama/Llama-3-8b-chat-hf`) and returns up to 512 tokens of chain-of-thought reasoning.

**Why so expensive?** It reflects reality — calling a hosted LLM costs real money per token. The 2.0 cost means the agent burns through 4% of its total episode budget on a single LLM call. It should only do this when genuinely necessary.

**Best for:** GPQA graduate-level science problems where factual retrieval isn't enough and actual reasoning is required.

**Graceful fallback:** If `TOGETHER_API_KEY` is not set, returns a clear error message instead of crashing. The agent learns to avoid this tool when it's unavailable.

**Tool routing note:** The environment exposes the canonical tool manifest at `GET /tools`, and tool dispatch normalizes missing-tool and tool-crash cases into explicit `ToolResult` errors. That keeps the OpenEnv-style contract stable even when a backing service is missing.

---

### `commit` — Cost: **0.0**

**What it does:** Submits the agent's answer for grading. This is always free — there's no penalty for committing, only for being wrong.

When commit is called:
1. The answer is extracted from `action.answer`
2. It's graded against the ground truth (see Section 7)
3. A reward is computed (see Section 5)
4. The context window is cleared
5. The episode advances to the next question

**Strategic note:** The agent should commit as soon as it's confident. Every extra tool call after reaching sufficient confidence is wasted budget.

---

## 5. The Reward Formula — Deep Dive

This is the core intellectual contribution of the environment. The reward has two components.

### Part 1: Step Reward (every tool call)

```
R_step = -tool_cost
```

Every time the agent calls a tool (including commit, which has cost 0), it gets a negative reward equal to the tool's cost. This creates constant pressure to be efficient.

| Tool | Step Reward |
|---|---|
| `ceramic_search` | -1.0 |
| `wiki_lookup` | -0.5 |
| `calculator` | -0.1 |
| `code_executor` | -0.3 |
| `llm_reason` | -2.0 |
| `commit` | 0.0 |

### Part 2: Commit Reward (on submit)

```
R_commit = base + bonus

base  = incorrect_reward + quality × (correct_reward − incorrect_reward)
      = -0.5 + quality × 1.5

bonus = η × γ × budget_remaining_ratio
      η = 1  if quality ≥ 0.5, else 0
      γ = 0.1  (efficiency weight)
      budget_remaining_ratio = remaining_budget / total_budget
```

Let's walk through what this means with real examples.

---

**Example A: Correct answer, lots of budget left**

The agent uses one `calculator` call (cost 0.1) and commits the right answer with 49.9 budget remaining out of 50.

```
quality = 1.0  (exact match)
base    = -0.5 + 1.0 × 1.5 = 1.0
η       = 1  (quality 1.0 ≥ threshold 0.5)
bonus   = 1 × 0.1 × (49.9/50) = 0.0998
R_step  = -0.1  (from the calculator call)

Total reward this question = -0.1 + 1.0 + 0.0998 = +0.9998
```

**Example B: Correct answer, lots of budget spent**

The agent uses three `ceramic_search` calls (cost 3.0 total) and commits right.

```
quality = 1.0
base    = 1.0
η       = 1
bonus   = 1 × 0.1 × (47/50) = 0.094
R_steps = -3.0  (three search calls)

Total = -3.0 + 1.0 + 0.094 = -1.906
```

Same correct answer, but much worse reward because of wasted tool calls.

**Example C: Wrong answer**

```
quality = 0.0  (wrong)
base    = -0.5 + 0.0 × 1.5 = -0.5
η       = 0  (quality 0.0 < threshold 0.5)  — no efficiency bonus
bonus   = 0

Total = R_steps + (-0.5)
```

**Example D: Partially correct answer (F1 = 0.6)**

```
quality = 0.6  (F1 score, close but not exact)
base    = -0.5 + 0.6 × 1.5 = 0.4
η       = 1  (0.6 ≥ 0.5)
bonus   = 1 × 0.1 × budget_ratio
```

Partial credit exists. This encourages the agent to try even when uncertain rather than just passing.

---

### Why this formula shape?

**The efficiency bonus gate (η):** The bonus only applies when quality ≥ 0.5. This prevents a degenerate strategy where the agent commits immediately with a random guess, earns a small efficiency bonus for having tons of budget left, and never actually tries to answer correctly.

**The linear quality scaling:** Rather than binary right/wrong, the agent gets gradual signal. Answering "Neil Armstrong" when the answer is "Neil Armstrong, Buzz Aldrin" gets partial credit. This makes learning easier because there's always a gradient to follow.

**The budget remaining ratio:** As budget drains, each correct answer is worth slightly less in efficiency bonus. This pushes the agent to be consistently frugal across all questions, not just on the last one.

---

## 6. The Datasets

Questions are drawn from four HuggingFace datasets, mixed according to `domain_mix` (default: 40/30/20/10):

### HotpotQA (40% of questions)
- **What it is:** Multi-hop Wikipedia questions that require connecting information from two or more sources.
- **Example:** *"What government position was held by the woman who portrayed Corliss Archer in the radio series?"*
- **Why it's hard:** A single search won't answer it. You need to find who played Corliss Archer, then look up that person's government role.
- **Best tool:** `ceramic_search` or `wiki_lookup` (multiple calls)
- **HuggingFace:** `hotpotqa/hotpot_qa`

### MATH (30% of questions)
- **What it is:** Competition mathematics problems at difficulty levels 3-5 (out of 5). These are AMC/AIME-style problems.
- **Example:** *"Find all real solutions to x³ - 6x² + 11x - 6 = 0"*
- **Why it's hard:** Requires algebraic reasoning, not just lookup. Level 5 problems stump most college students.
- **Best tool:** `calculator` for arithmetic steps, `code_executor` for complex algebra, `llm_reason` for symbolic reasoning
- **HuggingFace:** `DigitalLearningGmbH/MATH-lighteval`

### GPQA (20% of questions)
- **What it is:** Graduate-level questions in biology, chemistry, and physics, written by PhD students and vetted by domain experts.
- **Example:** *"Which of the following correctly describes the role of the TATA-binding protein in eukaryotic transcription initiation?"*
- **Why it's hard:** Even searching the exact question won't help — you need to understand the underlying science. This is where `llm_reason` earns its 2.0 cost.
- **Best tool:** `llm_reason`, possibly with a `ceramic_search` for context
- **HuggingFace:** `Idavidrein/gpqa` (gated — requires HF_TOKEN)

### HumanEval (10% of questions)
- **What it is:** Python programming tasks with a function signature and docstring. The agent must complete the function.
- **Example:** *"def is_palindrome(string: str) -> bool: \n    \"\"\"Test if a string is a palindrome.\"\"\""*
- **Why it's hard:** The agent needs to produce syntactically correct, logically correct code — and ideally run it to verify.
- **Best tool:** `code_executor` (write code, run it, verify output), `llm_reason` (generate code via LLM)
- **HuggingFace:** `openai/openai_humaneval`

---

## 7. Answer Grading

Grading happens in `env/answer_grading.py` when the agent calls `commit`.

### Step 1: Answer Extraction

The agent's answer string often contains extra text. We extract the actual answer using this priority chain:

1. **Strip markdown code fences** — remove ` ``` ` blocks
2. **Try JSON parsing** — if the answer looks like `{"answer": "Paris"}`, extract the `answer` field
3. **Prefix matching** — look for patterns like `"Answer: Paris"` or `"Final answer: Paris"`
4. **Last line fallback** — take the last non-empty line of the response

### Step 2: Normalization

Both the extracted answer and the ground truth are normalized the same way:
1. Lowercase everything
2. Remove articles (`a`, `an`, `the`)
3. Remove all punctuation
4. Split into tokens (words)

*Example:* `"The United States of America."` → `["united", "states", "america"]`

### Step 3: Exact Match (EM)

Are the normalized token lists identical?

```
predicted: "neil armstrong"    → ["neil", "armstrong"]
gold:      "Neil Armstrong"    → ["neil", "armstrong"]
EM = True ✓

predicted: "armstrong"         → ["armstrong"]
gold:      "Neil Armstrong"    → ["neil", "armstrong"]
EM = False ✗
```

### Step 4: Token F1

Token F1 measures overlap. It counts how many tokens appear in both the prediction and the ground truth, then computes precision and recall.

```
F1 = 2 × precision × recall / (precision + recall)

precision = common_tokens / predicted_tokens
recall    = common_tokens / gold_tokens
```

*Example:*
```
predicted: ["neil", "armstrong", "astronaut"]
gold:      ["neil", "armstrong"]

common  = ["neil", "armstrong"]  (2 tokens)
precision = 2/3 = 0.667
recall    = 2/2 = 1.0
F1 = 2 × 0.667 × 1.0 / (0.667 + 1.0) = 0.8
```

### Step 5: Quality Score

```python
quality = 1.0  if exact_match
quality = f1   otherwise
```

The quality score feeds directly into the reward formula. A perfect answer gets quality=1.0, a partial answer might get 0.4-0.8, and a completely wrong answer gets 0.0.

---

## 8. The Baselines

Three baseline policies ship with the environment. They don't learn — they follow fixed rules. Their purpose is to establish performance floors that a trained RL agent should beat.

### Random Tool (`baselines/random_tool.py`)

Picks a tool uniformly at random from the 5 non-commit tools each step. Commits after 3 steps with "I don't know."

**Expected behavior:** Wastes budget on expensive tools (llm_reason, ceramic_search) even for simple math problems. Gets ~20-30% accuracy by luck. Terrible budget efficiency.

**Why it exists:** Sets the absolute floor. Any RL agent that can't beat random is broken.

---

### Cheapest First (`baselines/cheapest_first.py`)

Calls tools in order of ascending cost: calculator (0.1) → code_executor (0.3) → wiki_lookup (0.5) → ceramic_search (1.0) → llm_reason (2.0). Commits after exhausting its call budget.

**Expected behavior:** Great budget efficiency. Terrible accuracy on HotpotQA and GPQA because it always tries the calculator first even on factual questions.

**Why it exists:** Shows that frugality alone isn't the answer. You need to route by domain, not just price.

---

### Domain Oracle (`baselines/oracle.py`)

Uses a hardcoded domain-to-tool mapping:
- HotpotQA → `ceramic_search`, then `wiki_lookup`
- MATH → `calculator`, then `llm_reason`
- GPQA → `llm_reason`, then `ceramic_search`
- HumanEval → `code_executor`, then `llm_reason`

Commits after exhausting its domain-specific sequence.

**Expected behavior:** Best accuracy of the three baselines. Still suboptimal because it doesn't adapt within a domain or manage budget across questions.

**Why it exists:** The performance ceiling for non-learning approaches. A trained RL agent should eventually exceed this by learning subtler patterns.

---

## 9. Where This Came From: SearchEconomicsEnv

ToolOrchestratorEnv is a direct generalization of [SearchEconomicsEnv](https://github.com/sharma-yash01/SearchEconomicsEnv), built by Yashaswi Sharma (University of Southern California) and Ceramic AI.

SearchEconomicsEnv posed a simpler version of the same question: given a fixed number of **search calls**, can an RL agent learn to answer HotpotQA questions efficiently? It used one tool (search), one dataset (HotpotQA), and Weitzman-style budget penalties.

The insight from that work was that agents could learn non-trivial search strategies — knowing when one search was enough versus when multiple hops were needed.

ToolOrchestratorEnv asks the harder question: **can the same principle scale to multiple tools and multiple domains?** Instead of "how many searches do I make?", the question becomes "which tool do I pick for this type of question, at this point in my budget?"

| | SearchEconomicsEnv | ToolOrchestratorEnv |
|---|---|---|
| Tools | 1 (search) | 6 (search, wiki, calc, code, LLM, commit) |
| Datasets | HotpotQA only | HotpotQA + MATH + GPQA + HumanEval |
| Budget unit | # of search calls | cost units per tool |
| Core challenge | How many searches? | Which tool, when? |
| Retrieval backend | Ceramic AI | Ceramic AI (shared) |

---

## 10. What a Trained Agent Should Learn

A well-trained agent should exhibit these behaviors:

**Domain routing.** When the question is math, skip search and go straight to calculator. When it's factual multi-hop, start with search. When it's graduate science, bite the bullet on llm_reason.

**Confidence-based committing.** If the calculator returned a clean number and the question was arithmetic, commit immediately. Don't spend another 0.5 on a Wikipedia lookup you don't need.

**Budget awareness.** In early questions with plenty of budget, it's okay to use ceramic_search. By question 8, with only 5 units left and 3 questions remaining, switch to calculator-only even for non-math questions.

**Failure recovery.** If the first tool call returns garbage (wrong article, irrelevant search results), try a different tool rather than committing a bad answer.

**These are the behaviors that baselines can't exhibit** — they require learning from feedback across thousands of episodes, which is exactly what RL provides.

---

## 11. File-by-File Reference

```
ToolOrchestratorEnv/
│
├── app.py
│   The FastAPI web server. Handles /reset, /step, /health, /tools, /web.
│   Multi-session: each /reset returns a session_id used in /step.
│   Lazily loads the dataset and exposes the canonical tool manifest.
│
├── openenv.yaml
│   Deployment spec for the OpenEnv competition framework.
│
├── Dockerfile
│   Builds the HuggingFace Space container. Runs uvicorn on port 8000.
│
├── env/
│   ├── environment.py
│   │   ToolOrchestratorEnvironment class. The main game loop.
│   │   reset() — starts episode, samples questions, zeroes budget.
│   │   step()  — validates action, dispatches tool, charges cost,
│   │             handles commit (grade + reward), manages episode end.
│   │   _make_obs() — assembles the observation dict from current state.
│   │   _sample_questions() — stratified sampling from dataset by domain.
│   │
│   ├── models.py
│   │   Pydantic types that define the agent-environment interface:
│   │   OrchestratorAction, ToolResult, OrchestratorObservation, OrchestratorState
│   │   TOOL_IDS — the canonical list of valid tool names.
│   │
│   ├── config.py
│   │   EnvConfig dataclass. All tuneable parameters:
│   │   total_budget, num_questions, tool_costs, domain_mix, reward weights.
│   │
│   ├── answer_grading.py
│   │   grade(predicted, gold) → (exact_match, f1, quality)
│   │   normalize_answer(), exact_match(), token_f1(), extract_answer()
│   │
│   └── reward.py
│       step_reward(tool_id, config) → negative cost
│       commit_reward(quality, budget_ratio, config) → composite score
│
├── ceramic/
│   └── client.py
│       CeramicClient — live API calls to https://api.ceramic.ai/search
│       FallbackCeramicClient — deterministic offline fake results
│       get_ceramic_client() — reads CERAMIC_API_KEY env var, returns right client
│
├── data/
│   └── loader.py
│       load_all(split, max_per_domain) — loads from HuggingFace datasets.
│       Falls back to synthetic questions if a dataset is unavailable.
│       Returns flat List[Dict] with 'domain' key on each item.
│
├── tools/
│   ├── runtime.py        Tool catalog, validation, and explicit dispatch
│   ├── __init__.py       build_tool_registry() + tool manifest helpers
│   ├── ceramic_search.py make_search_tool() factory wrapping CeramicClient
│   ├── wiki_lookup.py    Wikipedia REST API, first paragraph
│   ├── calculator.py     Safe AST-based math eval with comparisons
│   ├── code_executor.py  Sandboxed exec with blocked imports and dunder escapes
│   ├── llm_reason.py     Together AI API, graceful fallback
│   └── commit.py         Pass-through; grading is in environment.py
│
└── baselines/
    ├── random_tool.py    Uniform random tool selection
    ├── cheapest_first.py Always picks cheapest tool first
    └── oracle.py         Domain-aware heuristic routing
```

---

*This document describes the environment as implemented. For the blog post draft (with academic citations and related work), see `BLOG_PROMPT.md`.*
