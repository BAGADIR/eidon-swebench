# Eidon × DeepSeek-R1 on SWE-bench Verified

**Eidon** is a local-first repository intelligence engine. This repo runs it against the [SWE-bench Verified](https://www.swebench.com/) benchmark (500 real GitHub issues) using DeepSeek's thinking-mode model.

## How it works

**4-stage pipeline per task:**

```
Stage 1 — Encode    eidon analyze --fresh  →  .eidon/encoding (L0–L3 graph)
Stage 2 — Localize  deepseek-chat          →  top-5 files most likely to fix the bug
Stage 3 — Patch     deepseek-reasoner      →  think step-by-step → unified diff
Stage 4 — Repair    deepseek-reasoner      →  fix hunk-offset errors, re-apply
```

Eidon's `encoding` gives the model a **compressed L0–L3 architecture map** of the repo (purpose → interface → usage → full source) before it writes a single line of diff. This is the key differentiator vs. agents that search files at random.

## Key choices

| Decision | Value | Why |
|----------|-------|-----|
| Localize | `deepseek-chat` | Fast, structured JSON, no reasoning overhead |
| Patch | `deepseek-reasoner` | Thinking mode — explicit chain-of-thought on root cause |
| Repair | `deepseek-reasoner` | Reasons about why the hunk failed |
| Token budget | 32K (Eidon encoding) | Full architecture context |
| Patch tokens | 8K | Reasoner needs room for thinking tokens |

## Running locally

```bash
npm install -g eidoncore
eidon activate "$EIDON_LICENSE_KEY"
pip install -r swebench/requirements.txt
export DEEPSEEK_API_KEY=sk-...
python swebench/eidon_agent.py --tasks 10
```

## Running on GitHub Actions (500 tasks, 5 shards)

1. Fork this repo
2. Add secrets: `DEEPSEEK_API_KEY`, `EIDON_LICENSE_KEY`
3. Push — the workflow triggers automatically

**Estimated cost:** ~$7–10 for all 500 tasks at $0.28/1M input, $0.42/1M output.

## Submitting

```bash
pip install sb-cli
sb-cli submit --predictions predictions.json --split verified
```

## Leaderboard tag

`eidon-deepseek-reasoner`
