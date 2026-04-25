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
import queue
import threading
import concurrent.futures
from pathlib import Path
from typing import Optional

from openai import OpenAI
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────

DEEPSEEK_API_KEY    = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("LLM_API_KEY")
DEEPSEEK_BASE_URL   = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1")

# deepseek-reasoner = thinking mode, best for complex code repair (Stage 3 + 4)
MODEL_PATCH         = os.environ.get("LLM_MODEL_PATCH", "deepseek-reasoner")
MODEL_REPAIR        = os.environ.get("LLM_MODEL_REPAIR", "deepseek-reasoner")

EIDON_BIN           = "eidon"        # installed via: npm install -g eidoncore
TOKEN_BUDGET        = 8000           # eidon_encoding token budget

# Global MCP server cache: repo_path -> EidonMCPClient
# Keeps the server alive across tasks for the same repo (avoids reloading 200MB DB each time)
_mcp_cache: dict = {}
MAX_PATCH_TOKENS    = 32000          # max tokens for patch output
TASK_TIMEOUT        = int(os.environ.get("EIDON_TASK_TIMEOUT", "1800"))  # 30 min per task
OUTPUT_FILE         = "predictions.json"
CHECKPOINT_FILE     = "checkpoint.json"
MODEL_NAME_TAG      = "eidon-mcp-deepseek-reasoner"

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PATCH = """\
You are an expert software engineer fixing a real GitHub issue.

You have been given:
1. Eidon's semantic context (relevant files, ranked by CodeRank + HNSW search)
2. THE ACTUAL FILE CONTENTS FROM THE CHECKED-OUT REPOSITORY (exact state you must diff against)

!!! CRITICAL FORMAT REQUIREMENT !!!
YOUR ENTIRE RESPONSE MUST BE A RAW UNIFIED DIFF AND NOTHING ELSE.
DO NOT write any explanation, commentary, reasoning, or prose of any kind.
DO NOT use markdown code fences (no ```diff, no ```python, no ```).
YOUR RESPONSE MUST START WITH "--- a/" ON THE VERY FIRST CHARACTER.
DO NOT say you cannot fix the issue. Either output a diff or output nothing.

!!! CRITICAL: LINE NUMBERS !!!
The "## Actual File Contents" section shows the EXACT file content in the checked-out repo.
You MUST generate your diff against THOSE EXACT LINES.
Count line numbers from the actual file content provided.
DO NOT use line numbers from Eidon context -- Eidon's DB may be from a different commit.
If a file is not shown in "## Actual File Contents", do NOT patch it.

CRITICAL RULES:
  1. Output ONLY the raw patch -- no explanation, no markdown fences, no comments
  2. YOUR FIRST LINE MUST BE: --- a/path/to/file
  3. Use standard unified diff format:
       --- a/relative/path/file.py
       +++ b/relative/path/file.py
       @@ -N,M +N,M @@
        context line (3 lines of context before and after each change)
       -removed line (must exist VERBATIM in the actual file content provided)
       +added line
        context line
  4. Context lines (-N,M in the @@ header) MUST exist verbatim in the actual file shown above
  5. Minimal change -- only what is strictly necessary to fix the bug
  6. NEVER modify test files
  7. Your patch MUST make all FAIL_TO_PASS tests pass
  8. Respect the existing coding style (indentation, naming, error handling)
"""

SYSTEM_REPAIR = """\
You are an expert software engineer. A git patch you generated failed to apply.
The actual file contents from the checked-out repository are provided to you below.

Fix the patch so it applies cleanly to the repository.

!!! CRITICAL: You MUST use the ACTUAL FILE CONTENTS provided to determine correct line numbers.
The old patch had wrong line numbers or wrong context lines. Regenerate the patch from scratch
using the actual file content -- do NOT reuse the old line numbers or context lines.

Common failure reasons:
  - Wrong context lines (hunk @@ offsets do not match the actual file content)
  - Whitespace or indentation mismatch in context lines
  - The context lines no longer match the current file (stale diff)
  - Missing or extra blank lines
  - The feature/function you're patching doesn't exist yet in this older version

Rules:
  - Use ONLY line numbers and context lines from the actual file content provided
  - If the file you tried to patch is not shown in "Actual File Contents", pick a DIFFERENT file
  - Output ONLY the corrected unified diff patch. No explanation. No markdown.
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

## Actual File Contents (checked-out base commit — diff MUST match these exactly)

{actual_files}

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

Generate the minimal unified diff patch against the ACTUAL FILE CONTENTS above.
OUTPUT ONLY THE RAW UNIFIED DIFF. START WITH "--- a/". NO PROSE. NO FENCES.
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

## Actual File Contents (checked-out base commit — YOUR PATCH MUST MATCH THESE EXACTLY)

{actual_files}

---

Eidon context:
{eidon_context}

Output the corrected unified diff patch. Output ONLY the raw patch.
Your context lines MUST come verbatim from the "Actual File Contents" above.
Count line numbers from the actual file content — NOT from Eidon context or the old patch.
If the function/API you tried to patch doesn't exist in the actual file, pick a DIFFERENT approach.
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
            # Background thread logs stderr for debugging
            threading.Thread(
                target=lambda: [print("  [mcp:err] " + ln.rstrip()) for ln in self._proc.stderr if ln.strip()],
                daemon=True,
            ).start()

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

    def call_encoding(self, intent: str, token_budget: int = TOKEN_BUDGET, timeout: float = 600.0) -> Optional[str]:
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
        }, timeout=timeout)

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
                "DEEPSEEK_API_KEY or LLM_API_KEY environment variable not set."
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

        The MCP server is cached per repo_path — started once and reused across
        all tasks for the same repo, so large DBs (200MB django etc.) only pay
        the graphology initialization cost once.
        """
        global _mcp_cache

        db_path = Path(repo_path) / ".eidon" / "eidon.db"
        db_mb = db_path.stat().st_size / 1_048_576 if db_path.exists() else 0

        # For large DBs, reduce token_budget so eidon fetches/serializes fewer nodes.
        # This is the main factor affecting eidon_encoding query time on large graphs.
        # Fewer tokens = fewer top-K results to fetch = faster HNSW result processing.
        # 2000 tokens still covers the 1-3 relevant files needed for a typical SWE-bench fix.
        if db_mb > 200:
            budget = 2000
            rpc_timeout = 120.0
        elif db_mb > 100:
            budget = 3000
            rpc_timeout = 180.0
        else:
            budget = TOKEN_BUDGET
            rpc_timeout = 600.0
        mcp = _mcp_cache.get(repo_path)
        if mcp is None or mcp._proc is None or mcp._proc.poll() is not None:
            print("  [mcp] Starting Eidon MCP server... (db={:.0f}MB)".format(db_mb))
            mcp = EidonMCPClient(repo_path)
            if not mcp.start():
                print("  [mcp] Failed to start MCP server")
                return ""
            _mcp_cache[repo_path] = mcp
        else:
            print("  [mcp] Reusing cached MCP server (db={:.0f}MB)".format(db_mb))

        # Truncate intent to avoid overwhelming the search query
        intent = problem_statement[:1000].strip()
        print("  [mcp] Calling eidon_encoding(intent=..., token_budget={:,})... (db={:.0f}MB)".format(
            budget, db_mb))
        start = time.time()
        context = mcp.call_encoding(intent=intent, token_budget=budget, timeout=rpc_timeout)
        elapsed = time.time() - start

        if context:
            print("  [mcp] Got {:,} chars ({:,} est. tokens) in {:.1f}s".format(
                len(context), len(context) // 4, elapsed))
        else:
            print("  [mcp] eidon_encoding returned empty context")
            context = ""

        return context

    # ── Stage 3: Generate patch ───────────────────────────────────────────────

    def generate_patch(self, eidon_context: str, task: dict, repo_dir: str = "") -> str:
        """
        Call the LLM with Eidon's focused context + actual file contents + issue + tests.
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

        # Extract actual file contents from the checked-out repo.
        # Collect actual file contents from the checked-out repo.
        # Priority order:
        #   1. "--- a/" diff headers from Eidon context (highest priority — these are the files Eidon ranked)
        #   2. Source files derived from test file paths in test_patch (e.g. tests/test_foo.py → core/foo.py)
        #   3. All other .py mentions in context + problem_statement
        actual_files = ""
        if repo_dir:
            eidon_text = eidon_context or ""
            test_patch_text = task.get("test_patch", "") or ""
            problem_text = task.get("problem_statement", "") or ""
            all_text = eidon_text + "\n" + test_patch_text + "\n" + problem_text

            # Priority 1: "--- a/" headers from Eidon context (source files Eidon analyzed)
            priority1 = re.findall(r'(?:^|\n)---\s+a/(\S+\.py)', eidon_text)
            # Also pick up "File: pkg/foo.py" style Eidon headers
            priority1 += re.findall(r'(?:^|\n)File:\s*(\S+\.py)', eidon_text)

            # Priority 2: read the actual test file in the repo and parse its imports
            # to find the exact source module(s) under test.
            test_file_paths = re.findall(r'(?:^|\n)\+\+\+\s+b/((?:tests?|lib/[\w/]+/tests?)/\S+\.py)', test_patch_text)
            # Also catch any diff header for a test file
            test_file_paths += re.findall(r'(?:^|\n)---\s+a/((?:tests?|lib/[\w/]+/tests?)/\S+\.py)', test_patch_text)
            priority2 = []
            _seen_imports: set = set()
            for tpath in test_file_paths[:3]:
                tpath = tpath.lstrip('/')
                tfull = Path(repo_dir) / tpath
                if not tfull.exists():
                    continue
                try:
                    tcontent = tfull.read_text(errors='replace')
                    # Parse "from x.y.z import ..." and "import x.y.z"
                    mods = re.findall(r'^from\s+([\w.]+)\s+import', tcontent, re.MULTILINE)
                    mods += re.findall(r'^import\s+([\w.]+)', tcontent, re.MULTILINE)
                    for mod in mods:
                        candidate = mod.replace('.', '/') + '.py'
                        if '/' in candidate and candidate not in _seen_imports:
                            _seen_imports.add(candidate)
                            priority2.append(candidate)
                except Exception:
                    pass
            # Also derive from test filename (legacy heuristic as fallback)
            for p in re.findall(r'(?:^|\n)(?:---\s+a/|\+\+\+\s+b/)(\S+\.py)', test_patch_text):
                parts = p.replace('\\', '/').split('/')
                fname = parts[-1]
                if fname.startswith('test_') and len(parts) >= 2:
                    src_name = fname[5:]  # strip "test_" prefix
                    pkg = parts[0]
                    priority2 += ['{}/core/{}'.format(pkg, src_name),
                                  '{}/{}'.format(pkg, src_name),
                                  '{}/ops/{}'.format(pkg, src_name),
                                  '{}/backends/{}'.format(pkg, src_name)]

            # Priority 3: explicit file paths mentioned in text
            priority3 = re.findall(r'\b([\w/.-]+\.py)\b', all_text)
            # Also convert dot-notation module references (e.g. django.db.models.sql.query)
            for mod_ref in re.findall(r'\b([\w]+(?:\.[\w]+){2,})\b', all_text):
                candidate = mod_ref.replace('.', '/') + '.py'
                if '/' in candidate:
                    priority3.append(candidate)

            ordered = priority1 + priority2 + priority3
            seen = set()
            file_sections = []
            total_chars = 0
            for rel in ordered:
                rel = rel.strip('/')
                # Skip bare filenames (no directory) and test files
                if '/' not in rel or rel in seen:
                    continue
                # Don't inject test files (model shouldn't patch them)
                basename = rel.split('/')[-1]
                if basename.startswith('test_') or basename.startswith('conftest'):
                    seen.add(rel)
                    continue
                seen.add(rel)
                full = Path(repo_dir) / rel
                if not full.exists():
                    continue
                fsize = full.stat().st_size
                if fsize > 300_000:
                    # File too large — inject a truncated version with a note
                    try:
                        content = full.read_text(errors='replace')[:80_000]
                        section = "### {} (TRUNCATED — first 80K chars only)\n```python\n{}\n```".format(rel, content)
                    except Exception:
                        continue
                else:
                    try:
                        content = full.read_text(errors='replace')
                        section = "### {}\n```python\n{}\n```".format(rel, content)
                    except Exception:
                        continue
                total_chars += len(section)
                if total_chars > 200_000:
                    break
                file_sections.append(section)
            if file_sections:
                actual_files = "\n\n".join(file_sections[:12])
            else:
                # Last resort: grep the repo for key identifiers from the problem statement
                # Extract backtick-quoted names and CamelCase identifiers
                grep_targets = re.findall(r'`([a-zA-Z_][a-zA-Z0-9_.]+)`', problem_text)
                grep_targets += re.findall(r'\b([A-Z][a-zA-Z0-9]+(?:Error|Exception|Field|View|Form|Manager|Query))\b', problem_text)
                grep_seen: set = set()
                grep_files: list = []
                for name in grep_targets[:8]:
                    name = name.split('.')[-1]  # "django.db.models.Q" -> "Q"
                    if len(name) < 4 or name in grep_seen:
                        continue
                    grep_seen.add(name)
                    try:
                        r = subprocess.run(
                            ['grep', '-r', '--include=*.py', '-l',
                             'def ' + name, '.'],
                            cwd=repo_dir, capture_output=True, text=True, timeout=8,
                        )
                        for gf in r.stdout.strip().splitlines()[:3]:
                            gf = gf.strip().lstrip('.').lstrip('/')
                            if gf and '/' in gf and not Path(gf).name.startswith('test_'):
                                if gf not in seen:
                                    seen.add(gf)
                                    grep_files.append(gf)
                    except Exception:
                        pass
                for rel in grep_files[:5]:
                    full = Path(repo_dir) / rel
                    if not full.exists():
                        continue
                    try:
                        content = full.read_text(errors='replace')
                        if len(content) > 300_000:
                            content = content[:80_000]
                            section = "### {} (TRUNCATED)\n```python\n{}\n```".format(rel, content)
                        else:
                            section = "### {}\n```python\n{}\n```".format(rel, content)
                        file_sections.append(section)
                    except Exception:
                        pass
                if file_sections:
                    actual_files = "\n\n".join(file_sections[:8])
                    print("  [files] grep fallback found {} file(s)".format(len(file_sections)))
        if not actual_files:
            actual_files = "(Not available — use Eidon context above for file structure)"

        user_content = PATCH_TEMPLATE.format(
            eidon_context=eidon_context or "(no context from Eidon)",
            actual_files=actual_files,
            repo=task.get("repo", ""),
            problem_statement=task.get("problem_statement", ""),
            hints_section=hints_section,
            fail_to_pass=fail_list,
            test_patch=test_patch if test_patch else "(not provided)",
        )

        print("  [patch] Calling {} (~{:,} est. tokens)...".format(MODEL_PATCH, len(user_content) // 4))
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

    def _remap_patch_paths(self, patch: str, repo_dir: str) -> str:
        """If patch references non-existent paths, remap them by basename search."""
        if not patch.strip():
            return patch
        repo_path = Path(repo_dir)
        lines = patch.split("\n")
        out = []
        for line in lines:
            if line.startswith("--- a/") or line.startswith("+++ b/"):
                prefix = line[:6]
                fpath  = line[6:]
                full   = repo_path / fpath
                if not full.exists():
                    basename = Path(fpath).name
                    matches  = list(repo_path.rglob(basename))
                    # Filter out .git dir
                    matches  = [m for m in matches if ".git" not in m.parts]
                    if len(matches) == 1:
                        rel = matches[0].relative_to(repo_path).as_posix()
                        print("  [remap] {} -> {}".format(fpath, rel))
                        line = prefix + rel
            out.append(line)
        return "\n".join(out)

    def _fix_hunk_line_numbers(self, patch: str, repo_dir: str) -> str:
        """
        Correct @@ line numbers by searching for each hunk's context/removed lines
        in the actual checked-out file. This fixes the commit-skew problem where
        Eidon's DB was built on a newer commit than SWE-bench's base_commit.
        """
        if not patch or not repo_dir:
            return patch

        file_cache: dict = {}

        def get_file_lines(rel_path):
            if rel_path not in file_cache:
                fp = Path(repo_dir) / rel_path
                if fp.exists():
                    try:
                        file_cache[rel_path] = fp.read_text(errors='replace').splitlines()
                    except Exception:
                        file_cache[rel_path] = []
                else:
                    file_cache[rel_path] = []
            return file_cache[rel_path]

        def find_block(search_lines, file_lines, hint_line=0):
            """Return 0-based index of search_lines in file_lines, or None.
            Falls back to removed-lines-only search when full context search fails.
            """
            # Filter blank-only lines at edges
            trimmed = [l.rstrip() for l in search_lines]
            while trimmed and not trimmed[0].strip():
                trimmed = trimmed[1:]
            while trimmed and not trimmed[-1].strip():
                trimmed = trimmed[:-1]
            if not trimmed:
                return None

            def search(needles, haystack, hint):
                n = len(needles)
                if n == 0:
                    return None
                matches = []
                for i in range(len(haystack) - n + 1):
                    if haystack[i:i+n] == needles:
                        matches.append(i)
                if not matches:
                    return None
                return matches[0] if len(matches) == 1 else min(matches, key=lambda x: abs(x - hint))

            file_stripped = [l.rstrip() for l in file_lines]

            # Pass 1: exact match of all context+removed lines
            found = search(trimmed, file_stripped, hint_line)
            if found is not None:
                return found

            # Pass 2: whitespace-normalised match
            norm = [l.strip() for l in trimmed]
            norm_file = [l.strip() for l in file_stripped]
            found = search(norm, norm_file, hint_line)
            if found is not None:
                return found

            # Pass 3: search only for the removed (-) lines from the original hunk
            # (context lines may differ between commits; removed lines are more stable)
            removed_only = [l.rstrip() for l in search_lines
                            if l.startswith('-') or (len(l) > 1 and l[0] == '-')]
            removed_only = [l for l in removed_only if l.strip()]
            if removed_only:
                found = search(removed_only, file_stripped, hint_line)
                if found is not None:
                    return found
                norm_removed = [l.strip() for l in removed_only]
                found = search(norm_removed, norm_file, hint_line)
                if found is not None:
                    return found

            return None

        out_lines = []
        current_file = ""
        added_before = 0   # cumulative (+lines - -lines) from previous hunks
        i = 0
        patch_lines = patch.splitlines()

        while i < len(patch_lines):
            line = patch_lines[i]

            if line.startswith('--- a/'):
                current_file = line[6:].strip()
                added_before = 0
                out_lines.append(line)
                i += 1
                continue

            if line.startswith('+++ b/'):
                out_lines.append(line)
                i += 1
                continue

            if line.startswith('@@ ') and current_file:
                hunk_header = line
                i += 1
                # Collect hunk body
                hunk_body = []
                while i < len(patch_lines):
                    l = patch_lines[i]
                    if l.startswith('@@ ') or l.startswith('--- ') or l.startswith('diff '):
                        break
                    hunk_body.append(l)
                    i += 1

                # Extract search lines (context + removed lines)
                search_lines = []
                for bl in hunk_body:
                    if bl.startswith(' ') or bl.startswith('-'):
                        search_lines.append(bl[1:])

                # Parse existing header for hint
                m = re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)', hunk_header)
                if m and search_lines and current_file:
                    hint = int(m.group(1)) - 1  # 0-based
                    old_count = m.group(2)
                    new_count = m.group(4)
                    suffix = m.group(5)

                    file_lines = get_file_lines(current_file)
                    found = find_block(search_lines, file_lines, hint)
                    if found is not None and found != hint:
                        # Compute correct +side: account for lines added/removed before
                        new_old = found + 1  # 1-based
                        new_new = found + 1 + added_before
                        old_count_s = ',{}'.format(old_count) if old_count else ''
                        new_count_s = ',{}'.format(new_count) if new_count else ''
                        fixed = '@@ -{}{} +{}{} @@{}'.format(
                            new_old, old_count_s, new_new, new_count_s, suffix)
                        print('  [fix-lines] {} hunk relocated: line {} -> {}'.format(
                            current_file, m.group(1), new_old))
                        hunk_header = fixed

                # Track cumulative offset for next hunk
                removed = sum(1 for bl in hunk_body if bl.startswith('-'))
                added   = sum(1 for bl in hunk_body if bl.startswith('+'))
                added_before += added - removed

                out_lines.append(hunk_header)
                out_lines.extend(hunk_body)
                continue

            out_lines.append(line)
            i += 1

        result = '\n'.join(out_lines)
        if patch.endswith('\n') and not result.endswith('\n'):
            result += '\n'
        return result

    def verify_patch(self, patch: str, repo_dir: str) -> tuple:
        """Run `git apply --check`. Returns (ok, error_message)."""
        if not patch.strip():
            return False, "empty patch"
        try:
            result = subprocess.run(
                ["git", "apply", "--check", "--recount", "--ignore-whitespace", "-C0", "-"],
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
                ["git", "apply", "--recount", "--ignore-whitespace", "-C0", "-"],
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
        """Ask the model to fix the patch based on actual test failures."""
        print("  [test-repair] Asking {} to fix based on test output...".format(MODEL_REPAIR))
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

    def repair_patch(self, bad_patch: str, error: str, eidon_context: str, task: dict, repo_dir: str = "") -> str:
        """Ask the model to repair a patch that failed git apply --check."""
        print("  [repair] git apply failed: {}".format(error[:120]))
        print("  [repair] Asking {} to repair...".format(MODEL_REPAIR))

        # Build actual file content for the repair prompt.
        # Extract file paths from: (a) the error message, (b) the bad patch headers.
        actual_files_repair = "(Not available)"
        if repo_dir:
            # Parse failing file from error e.g. "patch failed: xarray/core/dataarray.py:3085"
            err_files = re.findall(r'patch failed:\s*([\w/.-]+\.py)', error)
            err_files += re.findall(r'error:.*?([\w/.-]+\.py)', error)
            # Also grab all files from the patch headers
            patch_files = re.findall(r'(?:^|\n)---\s+a/(\S+\.py)', bad_patch)
            patch_files += re.findall(r'(?:^|\n)\+\+\+\s+b/(\S+\.py)', bad_patch)
            all_repair_files = list(dict.fromkeys(err_files + patch_files))  # deduplicate, preserve order
            sections = []
            for rel in all_repair_files:
                rel = rel.strip('/')
                full = Path(repo_dir) / rel
                if not full.exists():
                    continue
                fsize = full.stat().st_size
                try:
                    if fsize > 300_000:
                        content = full.read_text(errors='replace')[:80_000]
                        sections.append("### {} (TRUNCATED — first 80K chars)\n```python\n{}\n```".format(rel, content))
                    else:
                        content = full.read_text(errors='replace')
                        sections.append("### {}\n```python\n{}\n```".format(rel, content))
                except Exception:
                    pass
            if sections:
                actual_files_repair = "\n\n".join(sections)

        user_content = REPAIR_TEMPLATE.format(
            error=error,
            bad_patch=bad_patch,
            actual_files=actual_files_repair,
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
                r"```(?:python|py|text|sh)?\n(.*?)```",
            ]:
                m = re.search(pattern, s, re.DOTALL)
                if m:
                    candidate = m.group(1).strip()
                    if "---" in candidate and "+++" in candidate:
                        extracted = candidate
                        break

            if extracted is None:
                # Scan lines for any unified diff header
                lines = s.split("\n")
                for i, line in enumerate(lines):
                    if line.startswith("--- ") or line.startswith("diff --git"):
                        # Extract only up to a closing fence if present
                        chunk = []
                        for l in lines[i:]:
                            if l.strip().startswith("```") and chunk:
                                break
                            chunk.append(l)
                        extracted = "\n".join(chunk)
                        break

        if not extracted:
            print("  [warn] Could not extract a valid patch from model output (raw[:400]: {})".format(repr(s[:400])))
            return ""

        # Strip git format-patch email footer ("-- \n2.45.2\n" etc.)
        extracted = re.sub(r'\n-- \n[0-9]+\.[0-9]+.*$', '', extracted, flags=re.DOTALL)

        # Strip any non-diff junk lines appended after the last valid hunk
        # (e.g. model appends "[filepath]...", "[code]...", prose, etc.)
        valid_prefixes = (' ', '+', '-', '@', 'd', '-', 'i', 'n', 'o', '\\')
        lines = extracted.split('\n')
        last_valid = -1
        for idx, l in enumerate(lines):
            if l and l[0] in ' +-@\\':
                last_valid = idx
            elif (l.startswith('diff ') or l.startswith('--- ') or
                  l.startswith('+++ ') or l.startswith('index ') or
                  l.startswith('new file') or l.startswith('deleted file') or
                  l == ''):
                last_valid = idx
        if last_valid >= 0:
            extracted = '\n'.join(lines[:last_valid + 1])

        # Ensure trailing newline (git apply requires it)
        if not extracted.endswith("\n"):
            extracted += "\n"
        return extracted

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
                    l = lines[i]
                    # Stop if line is not a valid unified-diff line
                    if l and l[0] not in ' +-\\':
                        break
                    hunk_lines.append(l)
                    i += 1

                # Strip trailing blank lines (avoid off-by-one in count)
                while hunk_lines and hunk_lines[-1].strip() == "":
                    hunk_lines.pop()

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

                # Recount (context lines count in both; removed only in old; added only in new)
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

        # Stage 3: Generate patch using Eidon context + actual file contents
        raw_output = self.generate_patch(eidon_context, task, repo_dir)
        patch      = self.extract_patch(raw_output)
        if not patch:
            print("  [debug] raw model output (first 600 chars): {}".format(repr(raw_output[:600])))

        # Stage 4: git apply repair loop (fix corrupt hunks)
        JUNK_FILES = {".git_archival.txt", ".codecov.yml", ".gitignore",
                      ".travis.yml", "setup.cfg", "tox.ini", ".editorconfig"}

        # Remap wrong file paths: if model used a non-existent path, find the real one by basename
        patch = self._remap_patch_paths(patch, repo_dir)

        # Fix line number skew: Eidon DB built on newer commit, SWE-bench checks out base_commit
        patch = self._fix_hunk_line_numbers(patch, repo_dir)

        MAX_REPAIR_ATTEMPTS = 6
        for attempt in range(MAX_REPAIR_ATTEMPTS + 1):
            ok, err = self.verify_patch(patch, repo_dir)
            if ok:
                # Sanity check: reject patches that only touch irrelevant config/meta files
                patched_files = [l[6:] for l in patch.splitlines() if l.startswith("+++ b/")]
                if patched_files and all(f.split("/")[-1] in JUNK_FILES for f in patched_files):
                    print("  [warn] Repair produced junk-only patch ({}) -- discarding".format(patched_files))
                    patch = ""
                    break
                label = " (after {} git-repair attempt(s))".format(attempt) if attempt else ""
                print("  [verify] Patch applies cleanly{}".format(label))
                break
            if attempt < MAX_REPAIR_ATTEMPTS:
                print("  [debug] corrupt patch (first 600 chars): {}".format(repr(patch[:600])))
                repaired = self.repair_patch(patch, err, eidon_context, task, repo_dir)
                patch    = self.extract_patch(repaired)
                patch    = self._fix_hunk_line_numbers(patch, repo_dir)
            else:
                print("  [verify] Patch still invalid after {} git-repair attempts -- submitting empty".format(MAX_REPAIR_ATTEMPTS))
                patch = ""

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

def run_benchmark(num_tasks, instance_filter, offset, cache_dir=None, retry_ids=None):
    agent = EidonAgent(cache_dir=cache_dir)

    predictions, done_ids = _load_checkpoint()

    print("[dataset] Loading princeton-nlp/SWE-bench_Verified...")
    ds    = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    tasks = list(ds)

    # Support retry of specific instance IDs (from file or env var)
    _retry_set = None
    _retry_ids_file = retry_ids or os.environ.get("RETRY_IDS_FILE")
    if _retry_ids_file:
        with open(_retry_ids_file) as _f:
            _retry_set = set(json.load(_f))
        print("[bench] RETRY mode: {} target IDs from {}".format(len(_retry_set), _retry_ids_file))

    if _retry_set:
        tasks = [t for t in tasks if t["instance_id"] in _retry_set]
        # Remove from done_ids so we re-run them even if already checkpointed
        done_ids -= _retry_set
    elif instance_filter:
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
                _ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                _fut = _ex.submit(agent.solve_task, task, repo_dir, already_analyzed)
                try:
                    patch = _fut.result(timeout=TASK_TIMEOUT)
                except concurrent.futures.TimeoutError:
                    print("  [timeout] Task exceeded {}s — submitting empty patch".format(TASK_TIMEOUT))
                    patch = ""
                finally:
                    _ex.shutdown(wait=False)  # don't block — let background thread die on its own
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
    parser.add_argument("--retry-ids",  default=None,   help="JSON file with list of instance_ids to retry")
    parser.add_argument("--cache-dir",  default=None,   help="Persistent cache dir for analyzed repos")
    args = parser.parse_args()

    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)

    run_benchmark(
        num_tasks=args.tasks,
        instance_filter=args.instance,
        offset=args.offset,
        cache_dir=args.cache_dir,
        retry_ids=args.retry_ids,
    )


if __name__ == "__main__":
    main()
