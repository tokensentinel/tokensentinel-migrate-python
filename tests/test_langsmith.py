"""Smoke tests for the LangSmith importer.

Mirrors :mod:`tests.test_helicone`'s mocking style: patch
``urllib.request.urlopen`` at the importer module's import site and feed
canned page payloads. Asserts the cursor-pagination + run_type filter +
dry-run wiring behave correctly. Not exhaustive — these are smoke tests
to catch CLI-wiring regressions, not a substitute for the in-importer
parsing unit tests that live under the same package.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

# Path-tweak preamble (mirrors test_helicone.py + test_langfuse.py).
_HERE = Path(__file__).resolve().parent
_MIGRATE_ROOT = _HERE.parent
_REPO_ROOT = _MIGRATE_ROOT.parent
_SDK_PATH = _REPO_ROOT / "sdk" / "python"
for path in (_SDK_PATH, _MIGRATE_ROOT):
    sp = str(path)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from tokensentinel_migrate.langsmith import migrate_langsmith  # noqa: E402

# ---------------------------------------------------------------------------
# Fake response helpers (copied from the helicone tests — keeping each
# test file self-contained beats a shared fixture for a 4-test suite).
# ---------------------------------------------------------------------------


def _fake_response(payload: Any, *, status: int = 200) -> Any:
    class _FakeResp:
        def __init__(self) -> None:
            body = json.dumps(payload).encode("utf-8") if payload is not None else b""
            self._body = body
            self.status = status
            self.headers: dict[str, str] = {}

        def read(self) -> bytes:
            return self._body

        def getcode(self) -> int:
            return self.status

        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *exc):  # type: ignore[no-untyped-def]
            return False

    return _FakeResp()


def _llm_run(
    *,
    run_id: str = "run_1",
    session_id: str | None = "sess_a",
    model: str = "claude-sonnet-4-6",
) -> dict:
    """One realistic LangSmith ``llm``-type run.

    Matches the API doc: a ``run_type`` of ``"llm"``, top-level token
    counts (the newer format the importer prefers), and ISO-8601 start /
    end timestamps with the ``Z`` suffix LangSmith ships.
    """
    return {
        "id": run_id,
        "run_type": "llm",
        "session_id": session_id,
        "name": model,
        "start_time": "2026-04-10T12:00:00Z",
        "end_time": "2026-04-10T12:00:01Z",
        "prompt_tokens": 80,
        "completion_tokens": 12,
        "extra": {
            "invocation_params": {
                "model": model,
                "_type": "chat-anthropic",
            }
        },
        "outputs": {"generations": [[{"message": {"content": "ok"}}]]},
    }


def _chain_run(*, run_id: str = "run_chain", session_id: str = "sess_a") -> dict:
    """A non-LLM run (run_type='chain'); the importer should drop these."""
    return {
        "id": run_id,
        "run_type": "chain",
        "session_id": session_id,
        "name": "RetrievalQA",
        "start_time": "2026-04-10T12:00:00Z",
        "end_time": "2026-04-10T12:00:02Z",
    }


# ---------------------------------------------------------------------------
# 1. Empty response -> zero-firing summary
# ---------------------------------------------------------------------------


def test_empty_response_returns_zero_summary() -> None:
    """``runs: []`` with ``cursors.next: null`` -> zero-everything summary."""

    def fake_urlopen(req, timeout=None):  # type: ignore[no-untyped-def]
        return _fake_response({"runs": [], "cursors": {"next": None}})

    with patch("tokensentinel_migrate.langsmith.urllib.request.urlopen", fake_urlopen):
        summary = migrate_langsmith(
            api_key="ls__test",
            tokensentinel_endpoint=None,
            tokensentinel_api_key=None,
            project="proj",
            since=None,
            dry_run=True,
        )

    assert summary.fetched_traces == 0
    assert summary.inferred_sessions == 0
    assert summary.fired_events == 0
    assert summary.events_by_type == {}
    assert summary.backfill_accepted == 0


# ---------------------------------------------------------------------------
# 2. Only run_type=='llm' is kept; chain / tool / retriever runs are dropped.
# ---------------------------------------------------------------------------


def test_only_llm_run_type_processed() -> None:
    """A page mixing LLM + chain runs -> only LLM rows seed sessions."""
    payload = {
        "runs": [
            _llm_run(run_id="llm_1", session_id="sess_a"),
            _chain_run(run_id="chain_1", session_id="sess_b"),
            _llm_run(run_id="llm_2", session_id="sess_a"),
        ],
        "cursors": {"next": None},
    }

    def fake_urlopen(req, timeout=None):  # type: ignore[no-untyped-def]
        return _fake_response(payload)

    with patch("tokensentinel_migrate.langsmith.urllib.request.urlopen", fake_urlopen):
        summary = migrate_langsmith(
            api_key="ls__test",
            tokensentinel_endpoint=None,
            tokensentinel_api_key=None,
            project="proj",
            since=None,
            dry_run=True,
        )

    # The importer fetched all three runs — but only the two ``llm`` ones
    # become CallRecords, which all collapse into the single ``sess_a``
    # session. ``sess_b`` (the chain run's session) never gets created.
    assert summary.fetched_traces == 3
    assert summary.inferred_sessions == 1


# ---------------------------------------------------------------------------
# 3. dry_run=True -> no backfill POST happens.
# ---------------------------------------------------------------------------


def test_dry_run_does_not_post_backfill() -> None:
    """With dry_run=True, the backfill leg is skipped entirely."""
    payload = {
        "runs": [_llm_run()],
        "cursors": {"next": None},
    }
    backfill_calls: list[str] = []

    def dispatching_urlopen(req, timeout=None):  # type: ignore[no-untyped-def]
        url = req.full_url
        if "smith.langchain.com" in url or "/runs/query" in url:
            return _fake_response(payload)
        # Backfill POSTs would route to the tokensentinel cloud — log so
        # the assertion can prove no such call happened.
        backfill_calls.append(url)
        return _fake_response({"accepted": 0, "rejected": 0})

    with patch(
        "tokensentinel_migrate.langsmith.urllib.request.urlopen",
        dispatching_urlopen,
    ):
        summary = migrate_langsmith(
            api_key="ls__test",
            tokensentinel_endpoint=None,
            tokensentinel_api_key=None,
            project="proj",
            since=None,
            dry_run=True,
        )

    assert backfill_calls == []
    assert summary.dry_run is True
    assert summary.backfill_accepted == 0
    assert summary.backfill_rejected == 0


# ---------------------------------------------------------------------------
# 4. Cursor pagination terminates correctly:
#    page 1 -> cursor "p2"; page 2 -> cursor null. urlopen called twice.
# ---------------------------------------------------------------------------


def test_cursor_pagination_terminates() -> None:
    """Two non-empty pages then a null cursor -> exactly two HTTP calls."""
    pages = [
        {"runs": [_llm_run(run_id="r1")], "cursors": {"next": "cursor_p2"}},
        {"runs": [_llm_run(run_id="r2")], "cursors": {"next": None}},
    ]
    page_iter = iter(pages)
    call_count = {"n": 0}

    def fake_urlopen(req, timeout=None):  # type: ignore[no-untyped-def]
        call_count["n"] += 1
        return _fake_response(next(page_iter))

    with patch("tokensentinel_migrate.langsmith.urllib.request.urlopen", fake_urlopen):
        summary = migrate_langsmith(
            api_key="ls__test",
            tokensentinel_endpoint=None,
            tokensentinel_api_key=None,
            project="proj",
            since=None,
            dry_run=True,
        )

    # Exactly two urlopen calls; both runs make it into the summary.
    assert call_count["n"] == 2
    assert summary.fetched_traces == 2
    assert summary.pages_fetched == 2
