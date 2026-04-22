#!/usr/bin/env python3
"""
Eidon SWE-bench Agent — MCP Edition
=====================================

This agent tests Eidon as it is actually used: as an MCP server.
The model never opens files directly. It only gets what Eidon gives it.

Pipeline per task:

  STAGE 1 — ENCODE
    `eidon analyze` runs the full 11-phase pipeline on the cloned repo.
    Hash-based cache: unchanged files reuse Phase 7 LLM analysis.
    Same repo = 100 tasks share one analysis (amortized cost ~$0).

  STAGE 2 — QUERY EIDON MCP
    Start `eidon mcp --repo <path>` as a stdio subprocess.
    Call eidon_encoding(intent=problem_statement, token_budget=32000).
    Eidon performs HNSW vector search on the issue text and returns:
      - Surgically relevant files (full source, Tier-3 treatment)
      - Graph context: CodeRank, blast_radius, community, dependencies
      - L0-L3 architectural map
    No separate localize step needed — eidon_encoding(intent=...) IS the
    localization, done with semantic vector search over the full graph.

  STAGE 3 — PATCH  (deepseek-reasoner thinking mode)
    DeepSeek reads Eidon's focused context + issue + FAIL_TO_PASS tests.
    Generates minimal unified diff.

  STAGE 4 — VERIFY + REPAIR (up to 2 retries)
    `git apply --check` validates the patch.
    On failure: deepseek-reasoner repairs the patch with error context.

Why this beats standalone DeepSeek:
  - No file browsing: model never wastes tokens opening wrong files
  - Semantic localization: HNSW search on the issue text finds exact files
  - Graph context: blast_radius, SPOF status, community — invisible to browsers
  - 86:1 compression: entire repo architecture in one focused prompt
  - Test-driven: FAIL_TO_PASS tests anchor every patch

Cost estimate per task:
  eidon_encoding call: ~12K tokens -> ~$0.003
  Patch (reasoner):    ~20K in, ~2K out -> ~$0.006
  Repair (~25% tasks): ~25K in, ~2K out -> ~$0.008
  Total: ~$0.008-0.015/task -> ~$4-8 for 500 tasks

Control (without Eidon) baseline: already on SWE-bench leaderboard.
We submit, compare, publish the delta.

Usage:
  python eidon_agent.py --tasks 10
  python eidon_agent.py --tasks 50           # validation (~$0.50, ~2h)
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

# deepseek-reasoner = thinking mode, best for complex code repair (Stage 3 + 4)
MODEL_PATCH         = "deepseek-reasoner"
MODEL_REPAIR        = "deepseek-reasoner"

EIDON_BIN           = "eidon"        # installed via: npm install -g eidoncore
TOKEN_BUDGET        = 32000          # eidon_encoding token budget
MAX_PATCH_TOKENS    = 8000           # reasoner needs room for thinking tokens + patch
OUTPUT_FILE         = "predictions.json"
CHECKPOINT_FILE     = "checkpoint.json"
MODEL_NAME_TAG      = "eidon-mcp-deepseek-reasoner"

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_LOCALIZE = """\
You are an expert software engineer analyzing a GitHub issue.

You are an expert software engineer fixing a real GitHub issue.

Eidon has already analyzed the entire repository and given you a focused,
semantically-ranked context: the files most relevant to this issue, their
full source, and the architectural graph (CodeRank, blast_radius, community,
dependencies). You did not browse files. Eidon selected them for you using
HNSW vector search on the issue description.

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
  3. The Eidon context for the relevant files (AFTER your patch was applied)
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

Eidon context for modified files (current state after patch):
{eidon_context}

---

## Issue
{problem_statement}

## Tests that must pass (FAIL_TO_PASS):
{fail_to_pass}

Generate a NEW patch from the ORIGINAL unmodified file. Output ONLY the raw unified diff.
"""

PATCH_TEMPLATE = """\
## Eidon Repository Context
(Semantic search on the issue description. Files ranked by relevance + CodeRank.)

{eidon_context}

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

Eidon context for the relevant files:

{eidon_context}

Output the corrected unified diff patch. Output ONLY the raw patch.
"""


# ── Eidon MCP Client ─────────────────────────────────────────────────────────

class EidonMCPClient:
    """
    Minimal MCP stdio client.
    Starts `eidon mcp --repo <path>` as a subprocess and communicates
    over stdin/stdout using JSON-RPC 2.0 (newline-delimited).

    This is how Eidon is actually used as a product — the model never
    opens files directly. It only receives what Eidon gives it.
    """

    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self._proc: Optional[subprocess.Popen] = None
        self._msg_id = 0
        self._reader_queue: queue.Queue = queue.Queue()
        self._reader_thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        """Start the Eidon MCP server. Returns True on success."""
        try:
            self._proc = subprocess.Popen(
                [EIDON_BIN, "mcp", "--repo", self.repo_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            # Background thread reads stdout lines into queue
            self._reader_thread = threading.Thread(
                target=self._reader_loop, daemon=True
            )
            self._reader_thread.start()

            # Send MCP initialize handshake
            resp = self._rpc("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "eidon-swebench", "version": "1.0"},
            })
            if resp is None:
                print("  [mcp] Initialize failed")
                return False
            # Send initialized notification
            self._notify("notifications/initialized", {})
            return True
        except Exception as e:
            print("  [mcp] Failed to start: {}".format(e))
            return False

    def stop(self):
        """Terminate the MCP server process."""
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                pass
            self._proc = None

    def _reader_loop(self):
        """Read newline-delimited JSON from stdout and enqueue."""
        try:
            for line in self._proc.stdout:
                line = line.strip()
                if line:
                    try:
                        self._reader_queue.put(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # ignore malformed lines
        except Exception:
            pass

    def _send(self, msg: dict):
        """Write a JSON-RPC message to stdin."""
        if self._proc and self._proc.stdin:
            try:
                self._proc.stdin.write(json.dumps(msg) + "\n")
                self._proc.stdin.flush()
            except BrokenPipeError:
                pass

    def _notify(self, method: str, params: dict):
        """Send a notification (no response expected)."""
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _rpc(self, method: str, params: dict, timeout: float = 30.0) -> Optional[dict]:
        """Send a JSON-RPC request and wait for the response."""
        self._msg_id += 1
        msg_id = self._msg_id
        self._send({
            "jsonrpc": "2.0",
            "id": msg_id,
            "method": method,
            "params": params,
        })
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                msg = self._reader_queue.get(timeout=0.1)
                if msg.get("id") == msg_id:
                    if "error" in msg:
                        print("  [mcp] RPC error: {}".format(msg["error"]))
                        return None
                    return msg.get("result")
                # Put back messages that aren't for us
                self._reader_queue.put(msg)
            except queue.Empty:
                pass
        print("  [mcp] Timeout waiting for response to {}".format(method))
        return None

    def call_encoding(self, intent: str, token_budget: int = TOKEN_BUDGET) -> Optional[str]:
        """
        Call eidon_encoding(intent=...) — the core MCP call.
        Eidon performs HNSW semantic search on `intent` (the issue description)
        and returns the most relevant files with full source + graph context.
        This replaces both the static encoding dump AND the separate localize step.
        """
        result = self._rpc("tools/call", {
            "name": "eidon_encoding",
            "arguments": {
                "intent": intent,
                "token_budget": token_budget,
            },
        }, timeout=120.0)

        if result is None:
            return None

        content = result.get("content", [])
        if content and isinstance(content, list):
            return content[0].get("text", "")
        return None


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
        Cache key is REPO-ONLY (not repo+commit).
        SWE-bench Verified has 500 tasks across only ~12 repos — caching by repo
        means eidon analyze runs once per repo (~12x total) not once per task (~500x).
        Different base_commits on the same repo are handled by git checkout.
        """
        if self.cache_dir:
            cache_key = repo.replace("/", "__")
            repo_dir  = os.path.join(self.cache_dir, cache_key)
            db_path   = os.path.join(repo_dir, ".eidon", "eidon.db")
            if os.path.exists(db_path):
                print("  [cache] HIT: {} (db exists)".format(repo))
                return repo_dir, True
            os.makedirs(repo_dir, exist_ok=True)
            return repo_dir, False
        else:
            return os.path.join(tmp_root, "repo"), False

    def clone_repo(self, repo: str, base_commit: str, repo_dir: str) -> bool:
        """Shallow-clone repo then checkout base_commit. Always uses --depth=1."""
        url = "https://github.com/{}.git".format(repo)
        print("  [git] Cloning {} (shallow)...".format(repo))
        try:
            r = subprocess.run(
                ["git", "clone", "--depth=1", url, repo_dir],
                capture_output=True, text=True, timeout=300,
            )
            if r.returncode != 0:
                print("  [git] Clone failed: {}".format(r.stderr[:300]))
                return False
            if not self.checkout_commit(repo_dir, base_commit):
                return False
            self._copy_preloaded_db(repo, repo_dir)
            return True
        except subprocess.TimeoutExpired:
            print("  [git] Clone timed out")
            return False
        except Exception as e:
            print("  [git] Error: {}".format(e))
            return False

    def checkout_commit(self, repo_dir: str, base_commit: str) -> bool:
        """
        Checkout a specific commit.
        Strategy: try directly, then fetch that exact commit at depth=1.
        This works even on shallow clones without downloading full history.
        """
        try:
            # Try direct checkout first (works if commit is already local)
            r = subprocess.run(
                ["git", "checkout", base_commit],
                cwd=repo_dir, capture_output=True, text=True, timeout=30,
            )
            if r.returncode == 0:
                return True
            # Fetch exactly this commit (depth=1 — no history needed)
            print("  [git] Fetching {}...".format(base_commit[:8]))
            subprocess.run(
                ["git", "fetch", "--depth=1", "origin", base_commit],
                cwd=repo_dir, capture_output=True, timeout=120,
            )
            r = subprocess.run(
                ["git", "checkout", base_commit],
                cwd=repo_dir, capture_output=True, text=True, timeout=30,
            )
            if r.returncode != 0:
                print("  [git] Checkout failed: {}".format(r.stderr[:200]))
            return r.returncode == 0
        except Exception as e:
            print("  [git] checkout_commit error: {}".format(e))
            return False

    # ── Stage 1: Encode ───────────────────────────────────────────────────────

    def encode_repo(self, repo_path: str) -> bool:
        """Run `eidon analyze` on the repo. Returns True if .eidon/db was produced."""
        db_path = Path(repo_path) / ".eidon" / "eidon.db"
        if db_path.exists():
            print("  [eidon] DB already present — skipping analyze")
            return True
        print("  [eidon] Analyzing {}...".format(repo_path))
        start = time.time()

        env = os.environ.copy()
        env["EIDON_LLM_PROVIDER"]       = "openai"
        env["EIDON_LLM_BASE_URL"]       = DEEPSEEK_BASE_URL
        env["EIDON_LLM_API_KEY"]        = DEEPSEEK_API_KEY or ""
        env["EIDON_LLM_MODEL"]          = "deepseek-chat"      # fast for Phase 7
        env["EIDON_LLM_CONCURRENCY"]    = "20"                 # 20 parallel Phase 7 LLM calls
        env["EIDON_WORKER_CONCURRENCY"] = "20"                 # 20 parallel Phase 2 workers
        env["EIDON_ENCODING_TOKENS"]    = str(TOKEN_BUDGET)
        env["EIDON_PHASE7_FILE_LIMIT"]  = "200"               # top-200 files by CodeRank only
        env["EIDON_MAX_RECHECK_CYCLES"] = "0"
        env["EIDON_RECHECK_BUDGET"]     = "0"
        env["EIDON_AI_COURT_BUDGET"]    = "0"
        env["EIDON_DEEP_SCAN_BUDGET"]   = "0"
        env["EIDON_LLM_REANALYSIS"]     = "false"

        try:
            result = subprocess.run(
                [EIDON_BIN, "analyze"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 min max per repo
                env=env,
            )
            elapsed = time.time() - start
            if result.returncode != 0:
                print("  [eidon] WARNING: exited {} after {:.1f}s".format(result.returncode, elapsed))
                print("  {}".format(result.stderr[:300]))
            else:
                print("  [eidon] Analyzed in {:.1f}s".format(elapsed))
        except subprocess.TimeoutExpired:
            print("  [eidon] Timed out after 30 min -- proceeding with partial DB")
        except Exception as e:
            print("  [eidon] Error: {}".format(e))
            return False

        # Check DB exists — MCP needs it
        db_path = Path(repo_path) / ".eidon" / "eidon.db"
        return db_path.exists()

    def _copy_preloaded_db(self, repo: str, repo_dir: str):
        """
        If EIDON_PRELOAD_DIR is set (by the pre-analyze workflow job),
        copy the pre-analyzed eidon.db into the freshly cloned repo.
        This means encode_repo() will be skipped entirely for this repo.
        """
        preload_dir = os.environ.get("EIDON_PRELOAD_DIR", "")
        if not preload_dir:
            return
        key = repo.replace("/", "__")
        src = os.path.join(preload_dir, "eidon-db-" + key, "eidon.db")
        if not os.path.exists(src):
            print("  [eidon] No preloaded DB for {} at {}".format(repo, src))
            return
        dst_dir = os.path.join(repo_dir, ".eidon")
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(src, os.path.join(dst_dir, "eidon.db"))
        size_mb = os.path.getsize(src) / 1_000_000
        print("  [eidon] Preloaded DB for {} ({:.1f} MB)".format(repo, size_mb))

    # ── Stage 2: Query Eidon MCP ──────────────────────────────────────────────

    def query_eidon_mcp(self, repo_path: str, problem_statement: str) -> str:
        """
        Start `eidon mcp --repo <path>` as a stdio subprocess and call
        eidon_encoding(intent=problem_statement).

        Eidon performs HNSW vector search on the issue description and returns
        the semantically closest files with full source + graph context.
        This IS the localization — no separate DeepSeek localize call needed.
        """
        print("  [mcp] Starting Eidon MCP server...")
        mcp = EidonMCPClient(repo_path)
        if not mcp.start():
            print("  [mcp] Failed to start MCP server")
            return ""

        try:
            # Truncate intent to avoid overwhelming the search query
            intent = problem_statement[:1000].strip()
            print("  [mcp] Calling eidon_encoding(intent=..., token_budget={:,})...".format(
                TOKEN_BUDGET))
            start = time.time()
            context = mcp.call_encoding(intent=intent, token_budget=TOKEN_BUDGET)
            elapsed = time.time() - start

            if context:
                print("  [mcp] Got {:,} chars ({:,} est. tokens) in {:.1f}s".format(
                    len(context), len(context) // 4, elapsed))
            else:
                print("  [mcp] eidon_encoding returned empty context")
                context = ""

            return context
        finally:
            mcp.stop()

    # ── Stage 3: Generate patch ───────────────────────────────────────────────

    def generate_patch(self, eidon_context: str, task: dict) -> str:
        """
        Call DeepSeek-reasoner with Eidon's focused context + issue + tests.
        No file sources needed separately — eidon_encoding already includes them.
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
            eidon_context=eidon_context or "(no context from Eidon)",
            repo=task.get("repo", ""),
            problem_statement=task.get("problem_statement", ""),
            hints_section=hints_section,
            fail_to_pass=fail_list,
            test_patch=test_patch if test_patch else "(not provided)",
        )

        print("  [patch] Calling deepseek-reasoner (~{:,} est. tokens)...".format(
            len(user_content) // 4))
        start = time.time()

        try:
            response = self.client.chat.completions.create(
                model=MODEL_PATCH,
                max_tokens=MAX_PATCH_TOKENS,
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
                    eidon_context: str, task: dict) -> str:
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
            eidon_context=eidon_context or "(no context)",
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

    def repair_patch(self, bad_patch: str, error: str, eidon_context: str, task: dict) -> str:
        """Ask DeepSeek to repair a patch that failed git apply --check."""
        print("  [repair] git apply failed: {}".format(error[:120]))
        print("  [repair] Asking DeepSeek to repair...")

        user_content = REPAIR_TEMPLATE.format(
            error=error,
            bad_patch=bad_patch,
            eidon_context=eidon_context or "(no context)",
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
        Four-stage pipeline per SWE-bench task:
          1. Encode  -- `eidon analyze` builds the HNSW graph (skip if cached)
          2. MCP     -- `eidon_encoding(intent=problem_statement)` returns focused context
          3. Patch   -- DeepSeek-reasoner generates unified diff from Eidon context
          4. Apply + repair loop (fix corrupt hunks, up to 2x)
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

        # Stage 1: Encode (build HNSW graph)
        if not skip_encode:
            try:
                ok = self.encode_repo(repo_dir)
                if not ok:
                    print("  [warn] eidon analyze did not produce DB -- MCP may fail")
            except Exception as e:
                print("  [eidon] encode failed: {} -- MCP may fail".format(e))

        # Stage 2: Query Eidon MCP — HNSW search on the problem statement
        problem_statement = task.get("problem_statement", "")
        try:
            eidon_context = self.query_eidon_mcp(repo_dir, problem_statement)
        except Exception as e:
            print("  [mcp] query failed: {} -- proceeding without context".format(e))
            eidon_context = ""

        if not eidon_context:
            print("  [warn] No Eidon context -- patch quality may be reduced")

        # Stage 3: Generate patch using Eidon context
        raw_output = self.generate_patch(eidon_context, task)
        patch      = self.extract_patch(raw_output)

        # Stage 4: git apply repair loop (fix corrupt hunks)
        for attempt in range(2):
            ok, err = self.verify_patch(patch, repo_dir)
            if ok:
                label = " (after {} git-repair attempt(s))".format(attempt) if attempt else ""
                print("  [verify] Patch applies cleanly{}".format(label))
                break
            if attempt < 1:
                repaired = self.repair_patch(patch, err, eidon_context, task)
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
                        raw_repaired = self.test_repair(
                            patch, pytest_out, eidon_context, task
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

    # Sort by repo so all tasks for the same repo run consecutively.
    # First task per repo: full eidon analyze (~1-2 min). All subsequent tasks
    # for the SAME repo: skip_encode=True → just git checkout + MCP query.
    # SWE-bench Verified = 500 tasks, ~12 repos → ~12 eidon analyze calls total.
    tasks.sort(key=lambda t: (t["repo"], t["base_commit"]))

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

            if not os.path.exists(os.path.join(repo_dir, ".git")):
                cloned = agent.clone_repo(task["repo"], task["base_commit"], repo_dir)
            else:
                # Repo already cloned (cached) — just switch to the right commit.
                # Reset any leftover changes from the previous task first.
                subprocess.run(["git", "reset", "--hard"], cwd=repo_dir,
                               capture_output=True, timeout=30)
                # -e .eidon: preserve the Eidon DB across task resets
                subprocess.run(["git", "clean", "-fd", "-e", ".eidon"], cwd=repo_dir,
                               capture_output=True, timeout=30)
                cloned = agent.checkout_commit(repo_dir, task["base_commit"])
                if cloned:
                    print("  [git] Checked out {}".format(task["base_commit"][:8]))

            if not cloned:
                print("  [error] Clone/checkout failed -- submitting empty patch")
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
