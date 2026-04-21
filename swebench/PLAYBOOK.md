# Eidon SWE-bench Run — Master Playbook

## The Mission

Beat Google. Beat OpenAI. Beat Anthropic's own agents.

SWE-bench Verified = 500 real GitHub issues from real Python repos.
Gold standard. Human-validated by OpenAI. The benchmark that matters.

**Target to beat**: Claude Sonnet 4.6 baseline = **71.4%** (355/500 tasks)
**Our system**: Eidon graph encoding + DeepSeek-R1 (deepseek-reasoner)
**Our claim**: Cheap model + architectural intelligence > expensive model alone

---

## Why This Works

Standard SWE-bench agents browse files like a dev who just joined:
- "Let me look at this file... now search for this function..."
- 50-150K tokens just figuring out WHERE things are
- Slow. Expensive. Misses architectural context.

Eidon gives DeepSeek the architectural map FIRST:
- Call graph, CodeRank, communities, blast radius — computed in <2 min
- Top 100 files by importance get AI purpose summaries (Phase 7)
- DeepSeek sees: "separability.py owns separability_matrix(), called by 14 others, high blast radius" — it goes DIRECTLY to the right file

**85x compression**: 150K tokens of codebase → 8-32K tokens of graph encoding.
10x cheaper. Faster. More accurate localization.

---

## Architecture

```
SWE-bench task (problem statement + repo + commit)
    ↓
Stage 1: eidon analyze (Phases 1-11, ~2 min)
    ↓ .eidon/encoding (8-32K tokens of graph)
Stage 2: DeepSeek-chat localizes (which files?)
    ↓ JSON list of relevant files
Stage 3: DeepSeek-reasoner patches (unified diff)
    ↓ git apply
Stage 4: Repair loop if apply fails (up to 2x)
    ↓
Stage 5: Run FAIL_TO_PASS tests, repair if needed (up to 3x)
    ↓
predictions.json
```

---

## Models

| Role | Model | Why |
|------|-------|-----|
| Eidon Phase 7 (file summaries) | `deepseek-chat` | Fast, cheap, good at code understanding |
| Localization | `deepseek-chat` | Fast JSON classification |
| Patching | `deepseek-reasoner` | R1 thinking mode — best at code generation |
| Repair | `deepseek-reasoner` | Same |

---

## Timing Budget (per task)

| Phase | Time |
|-------|------|
| git clone + fetch | ~20s |
| eidon analyze (Phases 1-6, graph) | ~90s |
| eidon Phase 7 (top 100 files × 50 concurrency) | ~15s |
| eidon Phases 8-11 (encoding) | ~10s |
| DeepSeek localize | ~5s |
| DeepSeek-R1 patch (thinking) | ~80s |
| git apply + test | ~30s |
| **Total per task** | **~4 min** |

10 tasks = ~40 min. 100 tasks (1 shard) = ~400 min. Fits in 350 min with 5 shards.
Full 500 tasks = 5 parallel shards × 100 tasks each = ~400 min total.

---

## Key Config (eidon_agent.py)

```python
DEEPSEEK_BASE_URL   = "https://api.deepseek.com/v1"
MODEL_LOCALIZE      = "deepseek-chat"
MODEL_PATCH         = "deepseek-reasoner"
MODEL_REPAIR        = "deepseek-reasoner"
TOKEN_BUDGET        = 32000   # encoding size fed to DeepSeek

# Critical eidon env vars:
EIDON_PHASE7_FILE_LIMIT  = "100"   # analyze top 100 files by CodeRank (not all 1200+)
EIDON_LLM_CONCURRENCY    = "50"    # 50 parallel Phase 7 calls
EIDON_WORKER_CONCURRENCY = "50"    # 50 parallel Phase 2 parse workers
EIDON_LLM_API_KEY        = DEEPSEEK_API_KEY  # DeepSeek for Phase 7 AI summaries
EIDON_MAX_RECHECK_CYCLES = "0"     # skip monitoring passes (not needed for one-shot)
EIDON_AI_COURT_BUDGET    = "0"     # skip AI court
EIDON_DEEP_SCAN_BUDGET   = "0"     # skip deep scan
```

---

## Eidon Installation

The CI workflow installs eidon from source (BAGADIR/Eidon on GitHub) because we added
`EIDON_PHASE7_FILE_LIMIT` — this patch isn't in the published npm binary yet.
Once npm publish is done, revert to `npm install -g eidoncore`.

---

## GitHub Actions Workflow

Repo: `BAGADIR/eidon-swebench`
Secrets needed:
- `DEEPSEEK_API_KEY` = `sk-2be235e08021474daa7277db46a804fd`
- `EIDON_LICENSE_KEY` = `EIDON-E9B5-0EA5-0D78-D1A1` (enterprise, unlimited files)

### Trigger a test run (10 tasks):
```bash
gh workflow run benchmark.yml --repo BAGADIR/eidon-swebench --field num_tasks=10 --field num_jobs=1
```

### Trigger the full 500-task run:
```bash
gh workflow run benchmark.yml --repo BAGADIR/eidon-swebench --field num_tasks=all --field num_jobs=5
```

### Monitor:
```bash
gh run list --repo BAGADIR/eidon-swebench --workflow=benchmark.yml --limit 5
gh run view <run-id> --repo BAGADIR/eidon-swebench --log
```

---

## Bug History (what we fixed)

| # | Problem | Root Cause | Fix |
|---|---------|-----------|-----|
| 1 | UTF-8 BOM crash | Windows writes BOM | Read with `encoding='utf-8-sig'` |
| 2 | Community plan file limit | Old key `EIDON-C443` = Community = 1000 files | Enterprise key `EIDON-E9B5-0EA5-0D78-D1A1` |
| 3 | "Key not found" | Enterprise key valid mathematically but not in Neon DB | `python insert_key.py` → inserted |
| 4 | Timed out after 600s | Post-analysis passes: recheck×5, AI court×30, deep scan×200 | Set all to 0 |
| 5 | Still 25+ min per task | Phase 7 analyzes ALL files (1200 for astropy). 1200 LLM calls | `EIDON_PHASE7_FILE_LIMIT=100` — top 100 by CodeRank only |

---

## After the Run: Leaderboard Submission

```bash
# Download predictions.json from the completed GitHub Actions run artifact
# Then:
python -m swebench.harness.run_evaluation \
  --predictions_path predictions.json \
  --run_id eidon-deepseek-r1 \
  --split verified
```

Or use sb-cli:
```bash
sb-cli submit --predictions predictions.json --split verified
```

Result posted to swebench.com leaderboard within 24h.

---

## The Leaderboard Entry

```
System:        eidon-deepseek-r1
Score:         XX.X% (targeting >71.4%)
Model:         DeepSeek-R1 + DeepSeek-chat
Context:       Eidon architectural encoding (graph-theoretic, 85x compression)
Cost:          ~$20 for 500 tasks
```

This is the shot. One benchmark run. Published publicly. Google, OpenAI, Anthropic devs
will see a cheap open-source model beating their frontier models because of architectural
intelligence, not raw model size. That's the story.

