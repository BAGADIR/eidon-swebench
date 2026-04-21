# Eidon SWE-bench Run — Execution Playbook

## What This Proves

SWE-bench Verified = 500 real GitHub issues from real Python repos (Django, scikit-learn,
Flask, Matplotlib, pytest, etc.). Human-validated by OpenAI. The gold standard benchmark.

Our claim: Eidon's graph encoding gives Claude a complete understanding of the codebase
BEFORE it touches a single file — so it fixes more bugs, correctly, with fewer API calls.

Current leaderboard baseline (mini-SWE-agent, no Eidon):
  - Claude Sonnet 4.6:  ~71.4% resolved
  - Claude Opus 4.6:    ~75.6% resolved
  - Best in world:      ~76.8% (Claude 4.5 Opus high reasoning)

Our target: Beat Claude Sonnet's baseline score by 3-5+ points using Eidon encoding.

---

## Cost Estimate

| Item | Cost |
|------|------|
| Claude API (50 validation tasks) | ~$2–5 |
| Claude API (full 500 tasks) | ~$20–40 |
| Compute (local laptop) | $0 |
| Docker evaluation (SWE-bench cloud) | Free (for leaderboard) |
| **Total** | **~$50 max** |

Why so cheap? Eidon compresses codebases 85x. Instead of Claude browsing
150K tokens of files per task, it gets 8-32K tokens of compressed graph. 10x cheaper.

---

## Setup (One Time)

### 1. Install Python dependencies
```bash
cd swebench/
pip install -r requirements.txt
```

### 2. Install SWE-bench CLI (for submission)
```bash
pip install swebench
# or: pip install sb-cli
```

### 3. Make sure Eidon is on PATH
```bash
# From root workspace:
npm install -g .
# Verify:
eidon --version
```

### 4. Set your Anthropic API key
```bash
# Windows PowerShell:
$env:ANTHROPIC_API_KEY = "sk-ant-..."

# Or add to .env (never commit this)
```

### 5. Verify setup
```bash
python eidon_agent.py --instance django__django-14580
```
This runs ONE task end-to-end. Should complete in ~90 seconds and cost <$0.10.

---

## Phase 1: Validation Run (50 tasks)

```bash
python eidon_agent.py --tasks 50
```

- Takes ~1.5 hours on a decent machine
- Costs ~$3–5
- Produces: `predictions.json` with 50 entries
- Produces: `checkpoint.json` (survives interruptions — resume anytime)

Submit to SWE-bench for scoring:
```bash
sb-cli submit --predictions predictions.json --split verified
```

You get back a score like "34/50 = 68%". Compare to baseline (Claude Sonnet = 71.4%
on full 500, but scores vary on subsets).

---

## Phase 2: Full Run (500 tasks)

Once validation confirms the approach works:
```bash
python eidon_agent.py --tasks all
```

- Takes ~8–12 hours (can be split across multiple sessions — checkpoint saves progress)
- Costs ~$20–40
- Produces: `predictions.json` with 500 entries

Submit:
```bash
sb-cli submit --predictions predictions.json --split verified
```

Official score posted to swebench.com leaderboard within 24h.

---

## What Gets Published

1. **Official leaderboard entry**: eidon-claude-sonnet-4-6 at X% on SWE-bench Verified
2. **Comparison**: X% vs Claude Sonnet baseline 71.4% (same model, same benchmark)
3. **Cost comparison**: $20 total vs ~$225 for standard agents (10x cheaper)
4. **Blog post**: "We ran SWE-bench Verified. Here's what happened."
5. **Send to boredabdel**: The exact link to the leaderboard entry

---

## Resuming After Interruption

The agent saves progress to `checkpoint.json` after every single task.
Just re-run the same command — it skips already-completed tasks automatically:

```bash
python eidon_agent.py --tasks 50  # Will skip already-done tasks
```

---

## Troubleshooting

**eidon analyze fails on a repo:**
- Check that Node.js ≥18 is installed
- Run `eidon analyze --fresh` manually in the repo dir to see the error
- The agent handles failures gracefully (skips encoding, still calls Claude)

**Claude returns no patch:**
- Check ANTHROPIC_API_KEY is set correctly
- Check you have credits on the account

**Repo clone fails:**
- GitHub rate limiting — add: `git config --global url.https://token@github.com/.insteadOf https://github.com/`
- Or set: `GH_TOKEN` env var

---

## The Narrative

Standard agents on SWE-bench browse files like a developer who just joined the company:
"Let me look at this file... now this one... now search for this function..."
They spend 50-150K tokens just figuring out where things are.

Eidon gives the agent the architectural map FIRST — like a senior engineer briefing
a junior before they start. The junior then goes directly to the right file, makes
the minimal change, and moves on.

That's the claim. SWE-bench will prove whether it's true.
