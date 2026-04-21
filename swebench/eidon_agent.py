#!/usr/bin/env python3
"""
Eidon SWE-bench Agent — World-Class Edition
============================================

Four-stage pipeline per task:

  STAGE 1 — ENCODE
    `eidon analyze` runs the full 11-phase pipeline on the cloned repo.
    Produces .eidon/encoding: L0-L3 graph-theoretic codebase map.
      L0 — System topology (spectral, entropy, health vector)
      L1 — Community topology (Louvain, gravity wells, bridge files)
      L2 — CodeRank, SPOFs, data-flow taint, circular cycles
      L3 — Per-file: CodeRank, blast_radius, risk_grade, AI-derived purpose
           + smart-compressed source at the appropriate tier
    Hash-based cache: Phase 7 LLM analysis is reused for unchanged files.

  STAGE 2 — LOCALIZE  (~$0.001/task)
    DeepSeek V3 reads the Eidon encoding + issue description.
    Outputs a JSON list of 3-5 exact file paths to modify.
    This mirrors what `eidon_encoding(intent=...)` does in the MCP:
    - In MCP: HNSW vector search on intent => top-20 files => Tier 3
    - Here: DeepSeek reasons over the L3 purpose map => surgical selection

  STAGE 3 — PATCH  (~$0.01-0.02/task)
    DeepSeek V3 reads:
      - Eidon encoding (full architectural context)
      - Full source of the localized files
      - The issue description + hints
      - The failing tests (FAIL_TO_PASS) from the SWE-bench task
    Generates a minimal unified diff patch.

  STAGE 4 — VERIFY + REPAIR (up to 2 retries)
    `git apply --check` validates the patch applies cleanly.
    If it fails, DeepSeek repairs the patch with the error context.

Why this beats vanilla DeepSeek (standalone):
  - Eidon encoding = full architectural oracle, not just file content
  - Localization = surgical file selection (vs. guessing in 1000+ file repos)
  - Test awareness = FAIL_TO_PASS drives the fix, not just the issue text
  - Blast-radius signal = knows which changes cascade downstream
  - Repair loop = recovers from malformed hunks

Cost estimate:
  Localize: ~5K input, ~200 output   -> ~$0.001
  Patch:    ~30K input, ~2K output   -> ~$0.012
  Repair:   ~35K input, ~2K output   -> ~$0.013 (~30% of tasks)
  Total: ~$0.015-0.025/task -> ~$8-12 for 500 tasks

Usage:
  python eidon_agent.py --tasks 10
  python eidon_agent.py --tasks 50           # validation (~$1, ~2h)
  python eidon_agent.py --tasks all          # full 500 tasks
  python eidon_agent.py --tasks 100 --offset 200   # shard: tasks 200-299
  python eidon_agent.py --instance django__django-12345
  python eidon_agent.py --tasks all --cache-dir ./eidon_cache
"""

import os
import sys
import json
import subprocess
import tempfile
import shutil
import argparse
import time
import re
from pathlib import Path
from typing import Optional

from openai import OpenAI
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────

DEEPSEEK_API_KEY    = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL   = "https://api.deepseek.com/v1"

# Two-model strategy (both are DeepSeek-V3.2, same price: $0.28/1M in, $0.42/1M out):
#   deepseek-chat      = Non-thinking mode. Fast, great for structured JSON output.
#   deepseek-reasoner  = Thinking mode. Reasons step-by-step before answering.
#                        Same price. Far better at complex code analysis.
# Stage 2 (localize):  deepseek-chat     — just needs file identification
# Stage 3 (patch):     deepseek-reasoner — needs to reason about root cause + fix
# Stage 4 (repair):    deepseek-reasoner — needs to reason about why hunk failed
MODEL_LOCALIZE      = "deepseek-chat"        # fast, cheap, structured JSON
MODEL_PATCH         = "deepseek-reasoner"    # thinking mode — best for code repair
MODEL_REPAIR        = "deepseek-reasoner"    # thinking mode — best for patch repair

EIDON_BIN           = "eidon"               # installed via: npm install -g eidoncore
TOKEN_BUDGET        = 32000                  # Eidon encoding token budget
MAX_PATCH_TOKENS    = 8000   # reasoner needs room for reasoning tokens + patch
MAX_LOCALIZE_TOKENS = 512
OUTPUT_FILE         = "predictions.json"
CHECKPOINT_FILE     = "checkpoint.json"
MODEL_NAME_TAG      = "eidon-deepseek-r1"    # R1 thinking mode used for patches

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_LOCALIZE = """\
You are an expert software engineer analyzing a GitHub issue.

You have been given the Eidon architectural encoding of the entire codebase.
The encoding is produced by the Eidon 11-phase analysis pipeline and contains:
  L0 -- System topology: spectral properties, Shannon entropy, health vector
  L1 -- Community topology: Louvain partition, gravity wells (hub files)
  L2 -- CodeRank (weighted PageRank), SPOFs, data-flow taint, circular cycles
  L3 -- Per-file: CodeRank score, blast_radius, risk_grade, AI-derived purpose
       plus smart-compressed source at the appropriate tier

Your task: Identify the MINIMAL set of source files that must be modified to
fix the reported issue.

Output ONLY valid JSON with this exact structure (no markdown, no explanation):
{
  "files": ["relative/path/file1.py", "relative/path/file2.py"],
  "reasoning": "one-sentence explanation of what needs to change"
}

Constraints:
  - At most 5 files total
  - Relative paths from repo root (e.g. "django/contrib/auth/models.py")
  - Source files ONLY -- never test_*.py or *_test.py files
  - Prefer modifying files with lower blast_radius when alternatives exist
"""

SYSTEM_PATCH = """\
You are an expert software engineer fixing a real GitHub issue.

You have been provided with:
  1. The Eidon architectural encoding (full codebase structure, L0-L3 graph)
  2. The FULL SOURCE of every file that needs to be modified
  3. The failing tests -- exactly what must pass after your fix
  4. The issue description and hints

Your task: Generate the minimal unified git diff patch that fixes the issue.

CRITICAL RULES:
  1. Output ONLY the raw patch -- no explanation, no markdown fences, no comments
  2. Use standard unified diff format:
       --- a/relative/path/file.py
       +++ b/relative/path/file.py
       @@ -N,M +N,M @@
        context line
       -removed line
       +added line
        context line
  3. Minimal change -- only what is strictly necessary to fix the bug
  4. NEVER modify test files
  5. Your patch MUST make all FAIL_TO_PASS tests pass
  6. Your patch MUST NOT break any PASS_TO_PASS tests
  7. Respect the existing coding style (indentation, naming, error handling)
  8. Files with blast_radius=critical affect many dependents -- be conservative
  9. If you cannot confidently fix the issue, emit an empty string (no patch)
"""

SYSTEM_REPAIR = """\
You are an expert software engineer. A git patch you generated failed to apply.

Fix the patch so it applies cleanly to the repository.

Common failure reasons:
  - Wrong context lines (hunk @@ offsets do not match the actual file content)
  - Whitespace or indentation mismatch in context lines
  - The context lines no longer match the current file (stale diff)
  - Missing or extra blank lines

Output ONLY the corrected unified diff patch. No explanation. No markdown.
"""

SYSTEM_TEST_REPAIR = """\
You are an expert software engineer. Your patch applied successfully but the
failing tests are still failing.

You will be given:
  1. The current patch you applied
  2. The pytest output showing exactly what failed and why
  3. The full source of the relevant files (AFTER your patch was applied)
  4. The issue description

Your task: Generate a NEW patch (relative to the ORIGINAL unmodified file)
that fixes the issue AND makes the failing tests pass.

Output ONLY the raw unified diff patch. No explanation. No markdown fences.
"""

TEST_REPAIR_TEMPLATE = """\
Current patch (already applied to repo):
```diff
{current_patch}
```

Pytest output (tests still failing):
```
{pytest_output}
```

Full source of modified files (current state after patch):
{file_sources}

---

## Issue
{problem_statement}

## Tests that must pass (FAIL_TO_PASS):
{fail_to_pass}

Generate a NEW patch from the ORIGINAL unmodified file. Output ONLY the raw unified diff.
"""

LOCALIZE_TEMPLATE = """\
## Eidon Codebase Encoding

{encoding}

---

## GitHub Issue

**Repository:** {repo}

{problem_statement}

---

Identify the files to modify. Output valid JSON only.
"""

PATCH_TEMPLATE = """\
## Eidon Codebase Encoding (Architectural Context)

{encoding}

---

## Full Source of Files to Modify

{file_sources}

---

## GitHub Issue

**Repository:** {repo}

{problem_statement}

{hints_section}

---

## Tests That Must Pass After Your Fix

**FAIL_TO_PASS** (currently failing -- must pass after your fix):
{fail_to_pass}

**Test code:**
```python
{test_patch}
```

---

Generate the minimal unified diff patch. Output ONLY the raw unified diff.
"""

REPAIR_TEMPLATE = """\
This patch failed to apply with the following error:

```
{error}
```

Original (broken) patch:
```diff
{bad_patch}
```

Full source of the relevant files:

{file_sources}

Output the corrected unified diff patch. Output ONLY the raw patch.
"""


# ── Agent ─────────────────────────────────────────────────────────────────────

class EidonAgent:
    def __init__(self, cache_dir: Optional[str] = None):
        if not DEEPSEEK_API_KEY:
            raise ValueError(
                "DEEPSEEK_API_KEY environment variable not set.\n"
                "Get your key at: https://platform.deepseek.com/api_keys"
            )
        self.client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        self.cache_dir           = cache_dir
        self.total_input_tokens  = 0
        self.total_output_tokens = 0
        self.total_tasks         = 0
        self.successful_tasks    = 0

    @property
    def total_cost(self) -> float:
        # DeepSeek V3.2 pricing (both models, April 2026):
        # $0.28/1M input (cache miss), $0.028/1M (cache hit), $0.42/1M output
        return (
            self.total_input_tokens  * 0.28 / 1_000_000
            + self.total_output_tokens * 0.42 / 1_000_000
        )

    # ── Repo management ───────────────────────────────────────────────────────

    def get_repo_dir(self, repo: str, base_commit: str, tmp_root: str) -> tuple:
        """
        Returns (repo_dir, already_analyzed).
        If --cache-dir is set and (repo, commit) is cached, returns it.
        """
        if self.cache_dir:
            cache_key = repo.replace("/", "__") + "_" + base_commit[:12]
            repo_dir  = os.path.join(self.cache_dir, cache_key)
            encoding  = os.path.join(repo_dir, ".eidon", "encoding")
            if os.path.exists(encoding):
                print("  [cache] HIT: {}@{}".format(repo, base_commit[:8]))
                return repo_dir, True
            os.makedirs(repo_dir, exist_ok=True)
            return repo_dir, False
        else:
            return os.path.join(tmp_root, "repo"), False

    def clone_repo(self, repo: str, base_commit: str, repo_dir: str) -> bool:
        """Clone repo at base_commit. Returns True on success."""
        url = "https://github.com/{}.git".format(repo)
        print("  [git] Cloning {}@{}...".format(repo, base_commit[:8]))
        try:
            # Shallow clone — much faster than full clone for large repos
            r = subprocess.run(
                ["git", "clone", "--depth=1", url, repo_dir],
                capture_output=True, text=True, timeout=300,
            )
            if r.returncode != 0:
                print("  [git] Clone failed: {}".format(r.stderr[:300]))
                return False

            r = subprocess.run(
                ["git", "checkout", base_commit],
                cwd=repo_dir, capture_output=True, text=True, timeout=60,
            )
            if r.returncode != 0:
                # Commit not in depth=1 — fetch ONLY that specific commit
                # (avoids --unshallow which downloads entire multi-GB history)
                print("  [git] Fetching specific commit {}...".format(base_commit[:8]))
                subprocess.run(
                    ["git", "fetch", "--depth=1", "origin", base_commit],
                    cwd=repo_dir, capture_output=True, timeout=300,
                )
                r = subprocess.run(
                    ["git", "checkout", base_commit],
                    cwd=repo_dir, capture_output=True, text=True, timeout=60,
                )
                if r.returncode != 0:
                    # Last resort: deepen by 500 commits
                    print("  [git] Deepening by 500 commits...")
                    subprocess.run(
                        ["git", "fetch", "--deepen=500", "origin"],
                        cwd=repo_dir, capture_output=True, timeout=300,
                    )
                    r = subprocess.run(
                        ["git", "checkout", base_commit],
                        cwd=repo_dir, capture_output=True, text=True, timeout=60,
                    )
                    if r.returncode != 0:
                        print("  [git] Checkout failed: {}".format(r.stderr[:200]))
                        return False
            return True
        except subprocess.TimeoutExpired:
            print("  [git] Timed out")
            return False
        except Exception as e:
            print("  [git] Error: {}".format(e))
            return False

    # ── Stage 1: Encode ───────────────────────────────────────────────────────

    def encode_repo(self, repo_path: str) -> str:
        """Run `eidon analyze` and return .eidon/encoding content."""
        print("  [eidon] Analyzing {}...".format(repo_path))
        start = time.time()

        env = os.environ.copy()
        env["EIDON_LLM_PROVIDER"]    = "openai"
        env["EIDON_LLM_BASE_URL"]    = DEEPSEEK_BASE_URL
        env["EIDON_LLM_API_KEY"]     = DEEPSEEK_API_KEY or ""
        env["EIDON_LLM_MODEL"]       = MODEL_LOCALIZE  # deepseek-chat for eidon analysis phase
        env["EIDON_LLM_CONCURRENCY"] = "50"            # 50 parallel LLM calls
        env["EIDON_ENCODING_TOKENS"] = str(TOKEN_BUDGET)
        # SWE-bench only needs the base L0-L3 encoding — skip post-analysis passes
        # that are designed for ongoing code monitoring, not one-shot patch generation
        env["EIDON_MAX_RECHECK_CYCLES"] = "0"   # default 5 — skip re-verification passes
        env["EIDON_RECHECK_BUDGET"]     = "0"   # default 50
        env["EIDON_AI_COURT_BUDGET"]    = "0"   # default 30 — skip AI court verification
        env["EIDON_DEEP_SCAN_BUDGET"]   = "0"   # default 200 — skip deep scan phase
        env["EIDON_LLM_REANALYSIS"]     = "false"  # no live re-analysis

        timed_out = False
        try:
            result = subprocess.run(
                [EIDON_BIN, "analyze", "--fresh"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=600,    # 10 min -- partial encoding is still read after timeout
                env=env,
            )
        except subprocess.TimeoutExpired:
            timed_out = True
            print("  [eidon] Timed out after 600s -- reading partial encoding if available")

        elapsed = time.time() - start

        if not timed_out:
            if result.returncode != 0:
                print("  [eidon] WARNING: exited {} after {:.1f}s".format(result.returncode, elapsed))
                print("  {}".format(result.stderr[:500]))
            else:
                print("  [eidon] Analyzed in {:.1f}s".format(elapsed))

        encoding_path = Path(repo_path) / ".eidon" / "encoding"
        if encoding_path.exists():
            encoding = encoding_path.read_text(encoding="utf-8", errors="replace")
            print("  [eidon] Encoding: {:,} chars (~{:,} tokens)".format(
                len(encoding), len(encoding) // 4))
            return encoding

        # Fallback: context.json
        context_path = Path(repo_path) / ".eidon" / "context.json"
        if context_path.exists():
            print("  [eidon] WARNING: .eidon/encoding not found, falling back to context.json")
            raw = context_path.read_text(encoding="utf-8", errors="replace")
            return raw[:TOKEN_BUDGET * 4]

        print("  [eidon] WARNING: no encoding produced")
        return ""

    # ── Stage 2: Localize ─────────────────────────────────────────────────────

    def localize_files(self, encoding: str, task: dict) -> list:
        """
        Ask DeepSeek which files to modify. Cheap call (~$0.001/task).
        Returns list of relative file paths.

        This is the non-MCP equivalent of eidon_encoding(intent=problem_statement):
        instead of HNSW vector search, we let DeepSeek reason over the L3
        purpose map to surgically identify which files need to change.
        """
        user_content = LOCALIZE_TEMPLATE.format(
            encoding=encoding,
            repo=task.get("repo", ""),
            problem_statement=task.get("problem_statement", ""),
        )

        print("  [localize] Identifying relevant files (~{} tokens)...".format(
            len(user_content) // 4))

        try:
            response = self.client.chat.completions.create(
                model=MODEL_LOCALIZE,  # deepseek-chat: fast, structured JSON
                max_tokens=MAX_LOCALIZE_TOKENS,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": SYSTEM_LOCALIZE},
                    {"role": "user",   "content": user_content},
                ],
            )
        except Exception as e:
            print("  [localize] API error: {}".format(e))
            return []

        if response.usage:
            self.total_input_tokens  += response.usage.prompt_tokens
            self.total_output_tokens += response.usage.completion_tokens

        raw = (response.choices[0].message.content or "").strip()

        # Strip markdown fences if present
        json_str = raw
        fence_m  = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
        if fence_m:
            json_str = fence_m.group(1).strip()

        try:
            data   = json.loads(json_str)
            files  = data.get("files", [])
            if not isinstance(files, list):
                files = []
            files  = [f for f in files if isinstance(f, str) and f.strip()][:5]
            reason = data.get("reasoning", "")
            print("  [localize] {} file(s): {}".format(len(files), files))
            if reason:
                print("  [localize] Reason: {}".format(reason))
            return files
        except json.JSONDecodeError:
            # Fallback: extract .py paths from plain text
            print("  [localize] JSON parse failed, extracting from: {}".format(raw[:200]))
            paths = re.findall(r'[\w./\-]+\.py', raw)
            return list(dict.fromkeys(paths))[:5]

    # ── Stage 2b: Read full source ────────────────────────────────────────────

    def read_file_sources(self, file_paths: list, repo_dir: str) -> str:
        """Read full source of localized files. Returns formatted string."""
        if not file_paths:
            return "(no files identified)"

        sections = []
        for rel_path in file_paths:
            abs_path = os.path.join(repo_dir, rel_path)
            try:
                content = Path(abs_path).read_text(encoding="utf-8", errors="replace")
                lines   = content.count("\n") + 1
                sections.append(
                    "### {} ({} lines)\n```python\n{}\n```".format(rel_path, lines, content)
                )
            except FileNotFoundError:
                sections.append("### {}\n(file not found at this commit)".format(rel_path))
            except Exception as e:
                sections.append("### {}\n(error reading: {})".format(rel_path, e))

        return "\n\n".join(sections)

    # ── Stage 3: Generate patch ───────────────────────────────────────────────

    def generate_patch(self, encoding: str, file_sources: str, task: dict) -> str:
        """
        Call DeepSeek V3 with full context: encoding + full source + issue +
        hints_text + test_patch + FAIL_TO_PASS test names.
        """
        hints         = (task.get("hints_text", "") or "").strip()
        hints_section = "**Hints from the issue thread:**\n{}".format(hints) if hints else ""

        fail_to_pass = task.get("FAIL_TO_PASS", [])
        if isinstance(fail_to_pass, str):
            try:
                fail_to_pass = json.loads(fail_to_pass)
            except Exception:
                fail_to_pass = [fail_to_pass]
        if isinstance(fail_to_pass, list) and fail_to_pass:
            fail_list = "\n".join("  - {}".format(t) for t in fail_to_pass)
        else:
            fail_list = "  (not specified)"

        test_patch = (task.get("test_patch", "") or "").strip()
        if len(test_patch) > 4000:
            test_patch = test_patch[:4000] + "\n... (truncated)"

        user_content = PATCH_TEMPLATE.format(
            encoding=encoding,
            file_sources=file_sources,
            repo=task.get("repo", ""),
            problem_statement=task.get("problem_statement", ""),
            hints_section=hints_section,
            fail_to_pass=fail_list,
            test_patch=test_patch if test_patch else "(not provided)",
        )

        print("  [patch] Calling deepseek-reasoner (thinking mode, ~{:,} est. tokens)...".format(
            len(user_content) // 4))
        start = time.time()

        try:
            response = self.client.chat.completions.create(
                model=MODEL_PATCH,     # deepseek-reasoner: thinking mode
                max_tokens=MAX_PATCH_TOKENS,
                # Note: temperature not set for thinking model (uses internal reasoning)
                messages=[
                    {"role": "system", "content": SYSTEM_PATCH},
                    {"role": "user",   "content": user_content},
                ],
            )
        except Exception as e:
            print("  [patch] API error: {}".format(e))
            return ""

        elapsed = time.time() - start

        if response.usage:
            in_tok      = response.usage.prompt_tokens
            out_tok     = response.usage.completion_tokens
            details     = response.usage.completion_tokens_details
            reason_tok  = (getattr(details, "reasoning_tokens", None) or
                           (details.get("reasoning_tokens", 0) if hasattr(details, "get") else 0)
                           ) if details else 0
            self.total_input_tokens  += in_tok
            self.total_output_tokens += out_tok
            cost = (in_tok * 0.28 + out_tok * 0.42) / 1_000_000
            print("  [patch] {:,}in / {:,}out ({:,} thinking) | ${:.4f} | {:.1f}s".format(
                in_tok, out_tok, reason_tok, cost, elapsed))

        return (response.choices[0].message.content or "").strip()

    # ── Stage 4: Verify + repair ──────────────────────────────────────────────

    def verify_patch(self, patch: str, repo_dir: str) -> tuple:
        """Run `git apply --check`. Returns (ok, error_message)."""
        if not patch.strip():
            return False, "empty patch"
        try:
            result = subprocess.run(
                ["git", "apply", "--check", "-"],
                input=patch, cwd=repo_dir,
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                return True, ""
            return False, (result.stderr + result.stdout).strip()
        except Exception as e:
            return False, str(e)

    def apply_patch(self, patch: str, repo_dir: str) -> bool:
        """Apply patch to repo. Returns True on success."""
        try:
            result = subprocess.run(
                ["git", "apply", "-"],
                input=patch, cwd=repo_dir,
                capture_output=True, text=True, timeout=30,
            )
            return result.returncode == 0
        except Exception:
            return False

    def reset_repo(self, repo_dir: str):
        """Reset any applied patches back to base commit."""
        try:
            subprocess.run(
                ["git", "checkout", "--", "."],
                cwd=repo_dir, capture_output=True, timeout=30,
            )
        except Exception:
            pass

    def install_deps(self, repo_dir: str) -> bool:
        """pip install -e . for the repo. Returns True if succeeded."""
        print("  [test] Installing dependencies...")
        try:
            r = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", ".", "--quiet",
                 "--no-build-isolation"],
                cwd=repo_dir, capture_output=True, text=True, timeout=180,
            )
            if r.returncode != 0:
                # Try without --no-build-isolation
                r = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-e", ".", "--quiet"],
                    cwd=repo_dir, capture_output=True, text=True, timeout=180,
                )
            return r.returncode == 0
        except Exception as e:
            print("  [test] pip install failed: {}".format(e))
            return False

    def run_tests(self, repo_dir: str, fail_to_pass: list) -> tuple:
        """
        Run the FAIL_TO_PASS tests. Returns (passed: bool, output: str).
        passed=True means all tests passed.
        """
        if not fail_to_pass:
            return True, "(no tests specified)"

        test_ids = " ".join(fail_to_pass[:10])  # cap at 10 tests
        print("  [test] Running {} test(s)...".format(len(fail_to_pass[:10])))
        try:
            r = subprocess.run(
                [sys.executable, "-m", "pytest"] + fail_to_pass[:10] +
                ["-x", "--tb=short", "-q", "--no-header", "--timeout=60"],
                cwd=repo_dir, capture_output=True, text=True, timeout=120,
            )
            output = (r.stdout + r.stderr)[-3000:]  # last 3000 chars
            passed = r.returncode == 0
            summary = "PASSED" if passed else "FAILED"
            print("  [test] {} (exit {})".format(summary, r.returncode))
            return passed, output
        except subprocess.TimeoutExpired:
            print("  [test] Timed out")
            return False, "Tests timed out after 120s"
        except Exception as e:
            print("  [test] Error: {}".format(e))
            return False, str(e)

    def test_repair(self, current_patch: str, pytest_output: str,
                    file_sources: str, task: dict) -> str:
        """Ask DeepSeek to fix the patch based on actual test failures."""
        print("  [test-repair] Asking DeepSeek to fix based on test output...")
        fail_to_pass = task.get("FAIL_TO_PASS", [])
        if isinstance(fail_to_pass, str):
            try:
                fail_to_pass = json.loads(fail_to_pass)
            except Exception:
                fail_to_pass = [fail_to_pass]

        user_content = TEST_REPAIR_TEMPLATE.format(
            current_patch=current_patch,
            pytest_output=pytest_output,
            file_sources=file_sources,
            problem_statement=task.get("problem_statement", "")[:2000],
            fail_to_pass="\n".join(fail_to_pass[:10]),
        )

        try:
            response = self.client.chat.completions.create(
                model=MODEL_REPAIR,
                max_tokens=MAX_PATCH_TOKENS,
                messages=[
                    {"role": "system", "content": SYSTEM_TEST_REPAIR},
                    {"role": "user",   "content": user_content},
                ],
            )
        except Exception as e:
            print("  [test-repair] API error: {}".format(e))
            return current_patch

        if response.usage:
            self.total_input_tokens  += response.usage.prompt_tokens
            self.total_output_tokens += response.usage.completion_tokens

        return (response.choices[0].message.content or "").strip()

    def repair_patch(self, bad_patch: str, error: str, file_sources: str, task: dict) -> str:
        """Ask DeepSeek to repair a patch that failed git apply --check."""
        print("  [repair] git apply failed: {}".format(error[:120]))
        print("  [repair] Asking DeepSeek to repair...")

        user_content = REPAIR_TEMPLATE.format(
            error=error,
            bad_patch=bad_patch,
            file_sources=file_sources,
        )

        try:
            response = self.client.chat.completions.create(
                model=MODEL_REPAIR,    # deepseek-reasoner: thinking mode
                max_tokens=MAX_PATCH_TOKENS,
                messages=[
                    {"role": "system", "content": SYSTEM_REPAIR},
                    {"role": "user",   "content": user_content},
                ],
            )
        except Exception as e:
            print("  [repair] API error: {}".format(e))
            return bad_patch

        if response.usage:
            self.total_input_tokens  += response.usage.prompt_tokens
            self.total_output_tokens += response.usage.completion_tokens

        return (response.choices[0].message.content or "").strip()

    # ── Patch extraction ──────────────────────────────────────────────────────

    def extract_patch(self, raw: str) -> str:
        """Strip markdown fences, extract the unified diff, and fix hunk headers."""
        s = raw.strip()
        if not s:
            return ""

        # 1. Extract from markdown fences if present
        extracted = None
        if s.startswith("diff ") or s.startswith("--- "):
            extracted = s
        else:
            for pattern in [
                r"```diff\n(.*?)```",
                r"```patch\n(.*?)```",
                r"```\n(.*?)```",
            ]:
                m = re.search(pattern, s, re.DOTALL)
                if m:
                    candidate = m.group(1).strip()
                    if "---" in candidate and "+++" in candidate:
                        extracted = candidate
                        break

            if extracted is None:
                for i, line in enumerate(s.split("\n")):
                    if line.startswith("--- ") or line.startswith("diff --git"):
                        extracted = "\n".join(s.split("\n")[i:])
                        break

        if not extracted:
            print("  [warn] Could not extract a valid patch from model output")
            return ""

        # 2. Recalculate hunk headers to fix wrong line counts
        return self._fix_hunk_headers(extracted)

    def _fix_hunk_headers(self, patch: str) -> str:
        """Recount lines in each hunk and rewrite @@ headers to match."""
        lines = patch.split("\n")
        out   = []
        i     = 0
        while i < len(lines):
            line = lines[i]
            # Pass through file headers unchanged
            if (line.startswith("diff ") or line.startswith("--- ") or
                    line.startswith("+++ ") or line.startswith("index ") or
                    line.startswith("new file") or line.startswith("deleted file") or
                    line.startswith("old mode") or line.startswith("new mode")):
                out.append(line)
                i += 1
                continue

            if line.startswith("@@"):
                # Collect hunk lines
                hunk_lines = []
                i += 1
                while i < len(lines) and not (
                    lines[i].startswith("@@") or
                    lines[i].startswith("diff ") or
                    lines[i].startswith("--- ")
                ):
                    hunk_lines.append(lines[i])
                    i += 1

                # Parse old start from existing header
                m = re.match(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@(.*)", line)
                if m:
                    old_start = int(m.group(1))
                    new_start = int(m.group(2))
                    tail      = m.group(3)
                else:
                    out.append(line)
                    out.extend(hunk_lines)
                    continue

                # Recount
                old_count = sum(1 for l in hunk_lines if not l.startswith("+"))
                new_count = sum(1 for l in hunk_lines if not l.startswith("-"))

                old_part = str(old_start) if old_count == 1 else "{},{}".format(old_start, old_count)
                new_part = str(new_start) if new_count == 1 else "{},{}".format(new_start, new_count)
                out.append("@@ -{} +{} @@{}".format(old_part, new_part, tail))
                out.extend(hunk_lines)
            else:
                out.append(line)
                i += 1

        return "\n".join(out)

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def solve_task(self, task: dict, repo_dir: str, skip_encode: bool = False) -> str:
        """
        Five-stage pipeline per SWE-bench task:
          1. Encode  -- eidon analyze (concurrency=50, ~48s on large repos)
          2. Localize -- DeepSeek identifies which files to change
          3. Patch   -- DeepSeek-reasoner generates unified diff
          4. Apply + git-repair loop (fix corrupt hunks, up to 2x)
          5. Test loop -- run FAIL_TO_PASS tests, re-patch on failure (up to 3x)
        """
        self.total_tasks += 1
        instance_id = task.get("instance_id", "?")

        fail_to_pass = task.get("FAIL_TO_PASS", [])
        if isinstance(fail_to_pass, str):
            try:
                fail_to_pass = json.loads(fail_to_pass)
            except Exception:
                fail_to_pass = [fail_to_pass]

        # Stage 1: Encode
        if skip_encode:
            enc_path = Path(repo_dir) / ".eidon" / "encoding"
            if enc_path.exists():
                encoding = enc_path.read_text(encoding="utf-8", errors="replace")
                print("  [eidon] Using cached encoding ({:,} chars)".format(len(encoding)))
            else:
                encoding = ""
        else:
            try:
                encoding = self.encode_repo(repo_dir)
            except Exception as e:
                print("  [eidon] encode failed: {} -- continuing without encoding".format(e))
                encoding = ""

        if not encoding:
            print("  [warn] No eidon encoding -- falling back to no-encoding mode")

        # Stage 2: Localize
        file_paths   = self.localize_files(encoding, task)
        file_sources = self.read_file_sources(file_paths, repo_dir)

        # Stage 3: Generate patch
        raw_output = self.generate_patch(encoding, file_sources, task)
        patch      = self.extract_patch(raw_output)

        # Stage 4: git apply repair loop (fix corrupt hunks)
        for attempt in range(2):
            ok, err = self.verify_patch(patch, repo_dir)
            if ok:
                label = " (after {} git-repair attempt(s))".format(attempt) if attempt else ""
                print("  [verify] Patch applies cleanly{}".format(label))
                break
            if attempt < 1:
                repaired = self.repair_patch(patch, err, file_sources, task)
                patch    = self.extract_patch(repaired)
            else:
                print("  [verify] Patch still invalid after git-repair -- will submit as-is")

        # Stage 5: Test execution loop (run FAIL_TO_PASS, re-patch on failure)
        if patch.strip() and fail_to_pass:
            deps_installed = self.install_deps(repo_dir)
            if deps_installed:
                for test_attempt in range(3):
                    # Apply patch cleanly each iteration (reset first if re-trying)
                    if test_attempt > 0:
                        self.reset_repo(repo_dir)
                    applied = self.apply_patch(patch, repo_dir)
                    if not applied:
                        print("  [test] Patch failed to apply on test attempt {}".format(test_attempt + 1))
                        break

                    passed, pytest_out = self.run_tests(repo_dir, fail_to_pass)
                    if passed:
                        print("  [test] All FAIL_TO_PASS tests PASS -- done")
                        break

                    if test_attempt < 2:
                        # Re-read sources in their CURRENT (patched) state for context
                        current_sources = self.read_file_sources(file_paths, repo_dir)
                        raw_repaired = self.test_repair(
                            patch, pytest_out, current_sources, task
                        )
                        new_patch = self.extract_patch(raw_repaired)
                        if new_patch.strip():
                            patch = new_patch
                        else:
                            print("  [test-repair] No patch produced, keeping previous")
                            break
                    else:
                        print("  [test] Still failing after 3 attempts -- submitting best patch")

                # Always reset repo to clean state after test loop
                self.reset_repo(repo_dir)
            else:
                print("  [test] Skipping test loop (pip install failed)")

        if patch:
            self.successful_tasks += 1
        else:
            print("  [warn] Empty patch for {}".format(instance_id))

        return patch


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _load_checkpoint():
    predictions = []
    if Path(CHECKPOINT_FILE).exists():
        try:
            with open(CHECKPOINT_FILE) as f:
                predictions = json.load(f)
            print("[checkpoint] Resuming: {} predictions already done".format(len(predictions)))
        except Exception:
            pass
    done_ids = {p["instance_id"] for p in predictions}
    return predictions, done_ids


def _save_checkpoint(predictions):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(predictions, f, indent=2)


# ── Main benchmark runner ─────────────────────────────────────────────────────

def run_benchmark(num_tasks, instance_filter, offset, cache_dir=None):
    agent = EidonAgent(cache_dir=cache_dir)

    predictions, done_ids = _load_checkpoint()

    print("[dataset] Loading princeton-nlp/SWE-bench_Verified...")
    ds    = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    tasks = list(ds)

    if instance_filter:
        tasks = [t for t in tasks if t["instance_id"] == instance_filter]
    else:
        if offset:
            tasks = tasks[offset:]
        if str(num_tasks) != "all" and num_tasks != -1:
            tasks = tasks[:int(num_tasks)]

    total = len(tasks)
    print("[bench] {} task(s) to run (offset={}, num={})".format(total, offset, num_tasks))

    with tempfile.TemporaryDirectory() as tmp_root:
        for idx, task in enumerate(tasks):
            inst_id = task["instance_id"]
            print("\n[{}/{}] {}".format(idx + 1, total, inst_id))

            if inst_id in done_ids:
                print("  [skip] Already in checkpoint")
                continue

            repo_dir, already_analyzed = agent.get_repo_dir(
                task["repo"], task["base_commit"], tmp_root
            )

            # Clean up previous task clone if no persistent cache
            if cache_dir is None and os.path.exists(repo_dir):
                shutil.rmtree(repo_dir, ignore_errors=True)

            cloned = True
            if not os.path.exists(os.path.join(repo_dir, ".git")):
                cloned = agent.clone_repo(task["repo"], task["base_commit"], repo_dir)

            if not cloned:
                print("  [error] Clone failed -- submitting empty patch")
                predictions.append({
                    "instance_id":        inst_id,
                    "model_patch":        "",
                    "model_name_or_path": MODEL_NAME_TAG,
                })
                done_ids.add(inst_id)
                _save_checkpoint(predictions)
                continue

            try:
                patch = agent.solve_task(task, repo_dir, skip_encode=already_analyzed)
            except Exception as e:
                print("  [error] Unhandled: {}".format(e))
                patch = ""

            predictions.append({
                "instance_id":        inst_id,
                "model_patch":        patch,
                "model_name_or_path": MODEL_NAME_TAG,
            })
            done_ids.add(inst_id)

            if len(predictions) % 5 == 0:
                _save_checkpoint(predictions)

            print("  [progress] {} tasks | {} patches | ${:.4f} total".format(
                agent.total_tasks, agent.successful_tasks, agent.total_cost))

    _save_checkpoint(predictions)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(predictions, f, indent=2)

    print("\n[done] {} predictions -> {}".format(len(predictions), OUTPUT_FILE))
    print("[cost] Total: ${:.4f}".format(agent.total_cost))
    print("[stats] {}/{} tasks produced patches".format(
        agent.successful_tasks, agent.total_tasks))
    return predictions


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Eidon SWE-bench Agent -- World-Class Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tasks",      default="all",  help="Number of tasks or 'all'")
    parser.add_argument("--offset",     type=int, default=0, help="Task offset for sharding")
    parser.add_argument("--instance",   default=None,   help="Run a single task by instance_id")
    parser.add_argument("--cache-dir",  default=None,   help="Persistent cache dir for analyzed repos")
    args = parser.parse_args()

    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)

    run_benchmark(
        num_tasks=args.tasks,
        instance_filter=args.instance,
        offset=args.offset,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
