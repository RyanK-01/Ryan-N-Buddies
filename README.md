# ShoppingButler

A conversational shopping agent for the TechJam Conversational E-Commerce
Search Challenge. ShoppingButler routes each turn onto one of two tracks — a
narrow, phrase-locked **Buying** track once the customer has stated concrete
constraints, or a wide, category-driven **Browsing** track while they
haven't — then re-scores the retrieved candidates by verbatim constraint-phrase
containment to close in on the customer's hidden target product within 10
turns.

The name reflects what it's built to do: attend closely to what the customer
actually says, ask only the follow-up questions that narrow the search, and
find the right product in as few turns as possible.

This README covers setup instructions and the short report (method, model
choice, limitations, and the latency/token/cost disclosure) required by
`docs/submission_rules.md`.

---

## Setup Instructions

**Requirements**

- Python 3.10+ (uses `from __future__ import annotations`)
- No third-party Python packages, and no `requirements.txt` — `agent.py`
  only uses the standard library (`json`, `math`, `os`, `re`, `sqlite3`,
  `time`, `urllib.request`, `dataclasses`, `pathlib`). Nothing to `pip
  install`.

**Optional: Ollama, for the LLM-assisted path**

ShoppingButler runs fully offline by default (see [Model Choice](#model-choice)
below), but its shipped configuration also calls a local **Ollama** server.
Skip this if you only want to reproduce the deterministic reference score.

1. Install Ollama: https://ollama.com/download
2. Pull the model the agent expects by default:
   ```bash
   ollama pull llama3.1:8b
   ```
3. Make sure the Ollama server is running (`ollama serve`, or it's already
   running as a background service after install).

**Configuration (environment variables, all optional)**

| Variable                 | Default                   | Purpose                                   |
|---------------------------|---------------------------|--------------------------------------------|
| `OLLAMA_HOST`             | `http://localhost:11434`  | Ollama server URL                          |
| `OLLAMA_MODEL`            | `llama3.1:8b`              | Model name to call                         |
| `OLLAMA_TIMEOUT`          | `30`                        | Per-call timeout (seconds), main LLM calls |
| `OLLAMA_EXTRACT_TIMEOUT`  | `8`                         | Per-call timeout (seconds), extraction calls |
| `LLM_EXTRACT`             | `1`                         | Turn-1 category recovery + reply-interpreter lane. Set `0` to disable. |
| `LLM_SESSION`             | `1`                         | Session-level clarification lane (C1/C2). Set `0` to disable. |
| `LLM_WALL_BUDGET_SECONDS` | `3600`                      | Run-wide LLM wall-clock ceiling. |

To point at a different model (e.g. one already pulled locally):

```bash
export OLLAMA_MODEL=llama3.1:8b   # or your own pulled model tag
```

To reproduce the fully deterministic reference score, turn every LLM flag
off:

```bash
LLM_EXTRACT=0 LLM_SESSION=0 python3 -m evaluator.local_evaluator
```

**Run**

From the repository root:

```bash
python3 -m evaluator.local_evaluator
```

This writes per-session results and aggregate metrics (Hit Rate@10, MRR,
MTTC) to `results.json`. See the repository root `README.md` for the full
Agent interface contract and scoring definitions.

---

## Method

- **Retrieval:** SQLite FTS5 BM25 over the frozen catalog — no embeddings.
  A dense-retrieval path was prototyped and measured to never win a
  candidate slot ahead of BM25 (0.000 contribution for a 25-minute embedding
  build and a per-turn network call), so it was removed; see the comment
  block at the top of `agent.py`.
- **Routing:** each turn is routed to a Buying track (verbatim constraints
  known — phrase-locked, narrower pool) or a Browsing track (no constraints
  yet — category-dominated query, deeper pool), since recall and precision
  needs differ between the two.
- **Scoring:** deterministic, document-frequency-weighted verbatim-phrase
  matching on top of BM25, so a common word like "cotton" can't earn the
  same trust as a rare, discriminative phrase. This fix alone lifted the
  deterministic public-set score from 0.8299 to 0.8367.
- **Question selection** (`ask_attribute`) is 100% deterministic throughout
  — a well-reasoned semantic attribute scored worse than the catch-all
  "other" against the evaluator's own constraint-classification priors, so
  there was nothing for an LLM to improve there.
- **Paraphrase-repair ladder:** if a customer reply doesn't match the regex
  templates, a multi-rung fallback (reworded-override detection, a
  deterministic catalog span probe, no-information detection, then a
  guarded LLM interpreter as a last resort) recovers the constraint phrase
  in case the private evaluator paraphrases replies more than the public
  one does. On the actual public simulator, the regex templates catch every
  reply, so the later rungs never execute — confirmed by diffing full
  session output with every LLM flag on vs. off (byte-identical, 0 LLM
  calls).

## Model Choice

- **No hosted/paid LLM API is used.** The agent talks to a **local Ollama**
  server over plain HTTP (`urllib.request` — no client library), calling
  **`llama3.1:8b`** by default.
- **The LLM is an assist, not a dependency.** It's used for three narrow,
  gated, capped roles: turn-1 category recovery, session-level
  clarification (fires at most once per session, only when the
  deterministic ladder finds nothing), and near-tie reranking (capped at 2
  calls/session). Every LLM output must pass a catalog-grounding guardrail
  (a verbatim substring of catalog data) before it's trusted.
- **Why local and gated, not always-on:** an earlier design called an LLM
  reranker unconditionally on every turn and was retired as a measured
  regression — `LLM_RERANK=1` with `qwen2.5:7b-instruct` scored 0.8643 vs.
  0.8686 deterministic (worse, for ~2,570 tokens/session), because an
  unbounded per-constraint LLM weight could outweigh a verbatim phrase hit
  and bury a correct rank-1 result. The two mechanisms that replaced it are
  both structurally bounded so they can't repeat that failure.
- **Network access:** required only for the optional LLM-assisted path, and
  only to `localhost` (or wherever `OLLAMA_HOST` points) — no external API
  key, no internet access, no third-party service. If organizer policy
  disables network access for final scoring, or Ollama isn't
  installed/reachable, the agent still runs correctly (see Limitations).

## Latency, Token Usage, and Cost Disclosure

Measured with `llama3.1:8b` on a live local Ollama, 200-session public-set
runs via `evaluator/local_evaluator.py`:

| Configuration                          | Tokens / 200 sessions | Wall-clock (LLM only) | TechnicalScore |
|-----------------------------------------|------------------------|------------------------|----------------|
| Deterministic only (`LLM_EXTRACT=0`)    | 0                      | 0s                     | 0.8367         |
| **Shipped default** (every LLM flag on) | ~42,200                | >10 min / ~216 calls (≈2.8s/call) | 0.8406 |

- **Per-call latency:** ≥~2.8s/call observed for `llama3.1:8b` locally
  (`OLLAMA_TIMEOUT=30s` / `OLLAMA_EXTRACT_TIMEOUT=8s` cap the worst case per
  call).
- **Run-wide budget:** `LLM_WALL_BUDGET_SECONDS=3600` (1 hour) governs total
  LLM wall-clock rather than a raw call count, so a long run degrades
  gracefully (falls back to deterministic scoring) instead of silently
  disabling LLM help partway through. Scaled from the public-set
  measurement above, an 800-session private run is estimated at ~35–45
  minutes of actual LLM wall-clock — a reasoned extrapolation, not a timed
  measurement at that scale.
- **Estimated model cost: $0.** `llama3.1:8b` runs locally via Ollama —
  there is no per-token API billing and no external service charge. The
  only cost is local compute/electricity while Ollama is running, which
  this disclosure does not attempt to price.
- **Net effect on score:** +0.0039 TechnicalScore over deterministic-only
  (hit rate unchanged at 0.955; MRR improves 0.646 → 0.659) for the
  ~42,200-token, minutes-not-seconds cost above. The full per-lane
  breakdown (including paraphrase-stress results) is in the "Shipped
  configuration" comment block at the top of `agent.py`.

## Limitations

- **Requires a local Ollama server for the LLM-assisted path.** Without it
  (or with the LLM flags turned off), the agent falls back to the
  deterministic BM25 + phrase-scoring path — every LLM call site is wrapped
  in its own try/except, degrading to deterministic ranking on any failure
  (unreachable host, timeout, malformed response) rather than stalling.
  After 3 cumulative LLM failures in a run, the LLM path latches off for
  every remaining session, so a network outage costs at most a few timed-out
  connection attempts, not a repeated per-turn stall.
- **No live credentials or internet access are required** — the only
  network call is to a local Ollama server, and the agent produces a
  complete, contract-shaped response every turn even fully offline.
- **The near-tie reranker (`LLM_TIEBREAK`) is a wash on its own terms:** it
  improves about as many sessions as it worsens, and is net-positive only
  because its wins are larger swings than its losses. It fixes none of the
  remaining misses where the target is buried deep in an already-retrieved
  pool or is genuinely outscored — that class of miss needs a better
  retrieval/scoring signal, not a reranker over the same candidates.
- **The LLM_WALL_BUDGET_SECONDS estimate for the 800-session private set is
  extrapolated, not measured** at that scale; re-verify against real
  private-scale timing (or the organizer's stated grading time budget)
  before a hard deadline, and prefer turning an individual `LLM_*` flag off
  over shrinking the budget if wall-clock turns out to be tight.
- **Paraphrase robustness is unverified against the actual private
  evaluator.** The paraphrase-repair ladder was stress-tested against a
  local, gitignored harness, not the organizer's private simulator, so its
  real-world paraphrase behavior is an informed guess, not a confirmed
  result.
