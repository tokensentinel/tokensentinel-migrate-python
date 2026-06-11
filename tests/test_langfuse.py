"""Smoke tests for the Langfuse importer.

These tests mirror :mod:`tests.test_helicone`'s mocking style — patch
``urllib.request.urlopen`` at the importer module's import site and feed
canned page payloads. The goal here isn't exhaustive coverage of the
Langfuse-specific parsing helpers; it's to assert the CLI -> importer ->
summary wiring works against realistic Langfuse Cloud response shapes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

# Ensure both the migrate package and the SDK are importable when the test
# is run via ``pytest tests/`` from the migrate root, even before
# ``pip install -e``. Same path-tweak preamble as test_helicone.py — copied
# rather than shared so each test file is self-contained.
_HERE = Path(__file__).resolve().parent
_MIGRATE_ROOT = _HERE.parent
_REPO_ROOT = _MIGRATE_ROOT.parent
_SDK_PATH = _REPO_ROOT / "sdk" / "python"
for path in (_SDK_PATH, _MIGRATE_ROOT):
    sp = str(path)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from tokensentinel_migrate.langfuse import migrate_langfuse  # noqa: E402

# ---------------------------------------------------------------------------
# Fake response helper (copied from test_helicone.py — keeping each test
# file self-contained beats a shared fixture for a 4-test smoke suite).
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


def _langfuse_page(
    *,
    traces: list[dict] | None = None,
    total_pages: int = 1,
) -> dict:
    """Wrap a list of traces in the ``{"data": ..., "meta": ...}`` envelope.

    Defaults to a single-page response with one totalItems claim so the
    importer's pagination loop terminates cleanly without further pages.
    """
    data = traces or []
    return {
        "data": data,
        "meta": {
            "totalItems": len(data),
            "totalPages": total_pages,
            "page": 1,
            "limit": 100,
        },
    }


def _trace_with_one_generation(
    *,
    trace_id: str = "tr_1",
    session_id: str | None = "sess_a",
    model: str = "claude-sonnet-4-6",
) -> dict:
    """Realistic minimal Langfuse trace carrying one GENERATION observation.

    Field names match the Langfuse Cloud API doc shape (``observations``
    array with ``type`` and ``usage`` fields). Token counts are deliberately
    non-zero so the resulting CallRecord has a valid burn estimate.
    """
    return {
        "id": trace_id,
        "sessionId": session_id,
        "timestamp": "2026-04-10T12:00:00Z",
        "observations": [
            {
                "id": f"obs_{trace_id}",
                "type": "GENERATION",
                "model": model,
                "startTime": "2026-04-10T12:00:00Z",
                "endTime": "2026-04-10T12:00:01Z",
                "usage": {"unit": "TOKENS", "input": 80, "output": 12},
                "output": {"content": []},
            }
        ],
    }


# ---------------------------------------------------------------------------
# 1. Empty response -> zero-firing summary
# ---------------------------------------------------------------------------


def test_empty_response_returns_zero_summary() -> None:
    """``data: []`` from Langfuse -> summary reports zero of everything."""

    def fake_urlopen(req, timeout=None):  # type: ignore[no-untyped-def]
        return _fake_response({"data": [], "meta": {"totalItems": 0, "totalPages": 0}})

    with patch("tokensentinel_migrate.langfuse.urllib.request.urlopen", fake_urlopen):
        summary = migrate_langfuse(
            public_key="pk-lf-test",
            secret_key="sk-lf-test",
            tokensentinel_endpoint=None,
            tokensentinel_api_key=None,
            project="proj",
            since=None,
            dry_run=True,
        )

    # Every counter on the summary should remain at its dataclass default.
    assert summary.fetched_traces == 0
    assert summary.inferred_sessions == 0
    assert summary.fired_events == 0
    assert summary.events_by_type == {}
    assert summary.backfill_accepted == 0
    assert summary.dry_run is True


# ---------------------------------------------------------------------------
# 2. Normal response with one GENERATION -> at least one CallRecord makes it
#    through to a session.
# ---------------------------------------------------------------------------


def test_normal_response_processes_observations() -> None:
    """A single trace with one GENERATION -> one session inferred."""
    page = _langfuse_page(traces=[_trace_with_one_generation()])

    def fake_urlopen(req, timeout=None):  # type: ignore[no-untyped-def]
        return _fake_response(page)

    with patch("tokensentinel_migrate.langfuse.urllib.request.urlopen", fake_urlopen):
        summary = migrate_langfuse(
            public_key="pk-lf-test",
            secret_key="sk-lf-test",
            tokensentinel_endpoint=None,
            tokensentinel_api_key=None,
            project="proj",
            since=None,
            dry_run=True,
        )

    # The importer pulled one trace and inferred one session from it.
    # fired_events may be 0 (the rules require multi-call patterns to
    # fire) but the call normalisation pipeline must have run.
    assert summary.fetched_traces == 1
    assert summary.inferred_sessions == 1


# ---------------------------------------------------------------------------
# 3. dry_run=True -> no backfill POST happens (and the summary stays clean).
# ---------------------------------------------------------------------------


def test_dry_run_does_not_post_backfill() -> None:
    """With dry_run=True, the backfill leg is skipped entirely."""
    page = _langfuse_page(traces=[_trace_with_one_generation()])
    backfill_calls: list[str] = []

    def dispatching_urlopen(req, timeout=None):  # type: ignore[no-untyped-def]
        url = req.full_url
        if "langfuse" in url or "/api/public/traces" in url:
            return _fake_response(page)
        # Anything else would be the backfill POST — log it so the
        # assertion can prove it stayed silent.
        backfill_calls.append(url)
        return _fake_response({"accepted": 0, "rejected": 0})

    with patch(
        "tokensentinel_migrate.langfuse.urllib.request.urlopen",
        dispatching_urlopen,
    ):
        summary = migrate_langfuse(
            public_key="pk-lf-test",
            secret_key="sk-lf-test",
            tokensentinel_endpoint=None,
            tokensentinel_api_key=None,
            project="proj",
            since=None,
            dry_run=True,
        )

    # Smoking gun: backfill module saw zero traffic.
    assert backfill_calls == []
    # Summary reports the dry-run posture explicitly.
    assert summary.dry_run is True
    assert summary.backfill_accepted == 0
    assert summary.backfill_rejected == 0


# ---------------------------------------------------------------------------
# 4. --langfuse-base-url is honoured (the override URL shows up in the
#    actual HTTP request the importer sends).
# ---------------------------------------------------------------------------


def test_custom_base_url_used_in_fetch() -> None:
    """A custom base_url is what the importer sends to urlopen."""
    captured_urls: list[str] = []
    page = _langfuse_page(traces=[])

    def fake_urlopen(req, timeout=None):  # type: ignore[no-untyped-def]
        captured_urls.append(req.full_url)
        return _fake_response(page)

    custom = "https://my.internal.langfuse"
    with patch("tokensentinel_migrate.langfuse.urllib.request.urlopen", fake_urlopen):
        migrate_langfuse(
            public_key="pk-lf-test",
            secret_key="sk-lf-test",
            tokensentinel_endpoint=None,
            tokensentinel_api_key=None,
            project="proj",
            since=None,
            base_url=custom,
            dry_run=True,
        )

    # At least one request went to our custom base URL — and no request
    # leaked to cloud.langfuse.com.
    assert any(url.startswith(custom) for url in captured_urls)
    assert not any("cloud.langfuse.com" in url for url in captured_urls)
