"""Tests for the Helicone importer.

Mocking strategy:

We monkey-patch ``urllib.request.urlopen`` at the modules where it's
*imported* (the importer's own module and the backfill module). Each test
provides a side-effect callable that reads the outgoing ``Request`` and
returns a fake response object — this lets us assert on both the requests
the importer sends (URLs, headers, body) and the responses we want it to
see (rate limits, error codes, paginated payloads).

We deliberately avoid pytest fixtures for the urlopen mock because we want
each test's expected URL flow to be visible inline — the test doubles as
documentation of what the importer hits.
"""

from __future__ import annotations

import io
import json
import sys
from datetime import datetime, timezone
from email.message import Message
from pathlib import Path
from typing import Any
from unittest.mock import patch
from urllib.error import HTTPError

# Ensure both the migrate package and the SDK are importable when the test
# is run via ``pytest tests/`` from the migrate root, even before
# ``pip install -e``. The SDK lives two directories up at
# ``../sdk/python``; we add it first because the migrate package depends on
# ``token_sentinel`` at import time.
_HERE = Path(__file__).resolve().parent
_MIGRATE_ROOT = _HERE.parent
_REPO_ROOT = _MIGRATE_ROOT.parent
_SDK_PATH = _REPO_ROOT / "sdk" / "python"
for path in (_SDK_PATH, _MIGRATE_ROOT):
    sp = str(path)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from tokensentinel_migrate import _backfill  # noqa: E402
from tokensentinel_migrate.cli import main as cli_main  # noqa: E402
from tokensentinel_migrate.helicone import (  # noqa: E402
    HeliconeImporter,
    _group_into_sessions,
    _infer_method,
    _infer_session_id,
    _to_call_record,
    migrate_helicone,
)

# ---------------------------------------------------------------------------
# Fake response helpers
# ---------------------------------------------------------------------------


def _fake_response(payload: Any, *, status: int = 200) -> Any:
    """Build a fake ``urllib.request.urlopen`` context-manager response.

    The real ``urlopen`` returns an object that:
      - is a context manager (``__enter__`` / ``__exit__``)
      - has ``.status`` (Py3.9+) or ``.getcode()`` (older)
      - has ``.read()`` returning bytes
      - has ``.headers`` mapping
    Our fake exposes all four so the same fake works against both
    ``_backfill._post_with_retry`` (which checks status) and
    ``HeliconeImporter._fetch_one_page`` (which calls ``read``).
    """

    class _FakeResp:
        def __init__(self):
            body = json.dumps(payload).encode("utf-8") if payload is not None else b""
            self._body = body
            self.status = status
            self.headers: dict[str, str] = {}

        def read(self) -> bytes:
            return self._body

        def getcode(self) -> int:
            return self.status

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    return _FakeResp()


def _make_http_error(status: int, *, retry_after: str | None = None) -> HTTPError:
    """Construct an ``HTTPError`` with a parsable ``Retry-After`` header.

    ``urllib.error.HTTPError.headers`` is a ``http.client.HTTPMessage`` (an
    email.message.Message subclass). We use ``email.message.Message``
    directly because it satisfies the ``.get(name)`` interface that the
    importer relies on.
    """
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return HTTPError(
        url="https://api.helicone.ai/v1/request/query",
        code=status,
        msg="rate limited" if status == 429 else "error",
        hdrs=headers,  # type: ignore[arg-type]
        fp=io.BytesIO(b""),
    )


def _helicone_request(
    *,
    request_id: str = "req_1",
    provider: str = "openai",
    model: str = "gpt-5",
    prompt_tokens: int = 50,
    completion_tokens: int = 5,
    latency_ms: float = 120.0,
    created_at: str = "2026-04-09T12:00:00Z",
    properties: dict | None = None,
    prompt: str = "Classify this as positive or negative",
) -> dict:
    """Synthesize a Helicone ``request`` dict for tests.

    The defaults produce a small, classification-shaped prompt routed at a
    frontier model — ie, exactly the shape ``model_misroute`` is built to
    catch. Tests that want a different shape pass overrides.
    """
    return {
        "request_id": request_id,
        "provider": provider,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "latency_ms": latency_ms,
        "created_at": created_at,
        "properties": properties or {},
        "prompt": prompt,
        "body": {"messages": [{"role": "user", "content": prompt}]},
    }


# ---------------------------------------------------------------------------
# 1. Pagination
# ---------------------------------------------------------------------------


def test_pagination_three_pages_all_fetched():
    """Three pages of 100, 100, 47 — importer fetches all 247 traces."""
    pages = [
        [_helicone_request(request_id=f"req_{i}") for i in range(100)],
        [_helicone_request(request_id=f"req_{i}") for i in range(100, 200)],
        [_helicone_request(request_id=f"req_{i}") for i in range(200, 247)],
    ]
    page_iter = iter(pages)

    def fake_urlopen(req, timeout=None):
        return _fake_response(next(page_iter))

    importer = HeliconeImporter(api_key="sk-helicone-test")
    with patch("tokensentinel_migrate.helicone.urllib.request.urlopen", fake_urlopen):
        all_pages = list(importer.iter_pages())

    flat = [r for page in all_pages for r in page]
    assert len(flat) == 247
    # Pages are yielded in order: 100, 100, 47.
    assert [len(p) for p in all_pages] == [100, 100, 47]


# ---------------------------------------------------------------------------
# 2. ``since`` cutoff respected
# ---------------------------------------------------------------------------


def test_since_cutoff_drops_older_traces():
    """Anything older than ``since`` is filtered; pagination stops on cutoff breach."""
    page1 = [
        _helicone_request(request_id="r1", created_at="2026-04-15T00:00:00Z"),
        _helicone_request(request_id="r2", created_at="2026-04-14T00:00:00Z"),
        _helicone_request(request_id="r3", created_at="2026-04-08T00:00:00Z"),  # older
    ]
    pages = [page1, []]  # second page won't be fetched once cutoff breached
    page_iter = iter(pages)

    def fake_urlopen(req, timeout=None):
        return _fake_response(next(page_iter))

    importer = HeliconeImporter(api_key="sk-helicone-test")
    cutoff = datetime(2026, 4, 9, tzinfo=timezone.utc)
    with patch("tokensentinel_migrate.helicone.urllib.request.urlopen", fake_urlopen):
        all_pages = list(importer.iter_pages(since=cutoff))

    flat = [r for page in all_pages for r in page]
    # Only r1 and r2 survive — r3 is older than 2026-04-09.
    assert len(flat) == 2
    ids = {r["request_id"] for r in flat}
    assert ids == {"r1", "r2"}


# ---------------------------------------------------------------------------
# 3. 429 + Retry-After triggers backoff
# ---------------------------------------------------------------------------


def test_429_with_retry_after_triggers_backoff():
    """A 429 response with ``Retry-After: 5`` makes the importer sleep for 5s and retry."""
    call_state = {"calls": 0}
    sleep_calls: list[float] = []
    page = [_helicone_request(request_id="r1")]

    def fake_urlopen(req, timeout=None):
        call_state["calls"] += 1
        if call_state["calls"] == 1:
            raise _make_http_error(429, retry_after="5")
        return _fake_response(page)

    def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    importer = HeliconeImporter(api_key="sk-helicone-test")
    with (
        patch("tokensentinel_migrate.helicone.urllib.request.urlopen", fake_urlopen),
        patch("tokensentinel_migrate.helicone.time.sleep", fake_sleep),
    ):
        all_pages = list(importer.iter_pages())

    # We saw two urlopen calls (the 429 then the retry success) and one
    # sleep, of exactly 5.0 seconds (the Retry-After value).
    assert call_state["calls"] == 2
    assert sleep_calls == [5.0]
    assert importer.rate_limit_pauses == 1
    flat = [r for page in all_pages for r in page]
    assert len(flat) == 1


# ---------------------------------------------------------------------------
# 4. Session inference: Helicone-Session-Id wins
# ---------------------------------------------------------------------------


def test_session_id_helicone_property_wins():
    """When ``Helicone-Session-Id`` is set, it beats ``session_id`` and ``request_id``."""
    req = _helicone_request(
        request_id="ignored",
        properties={
            "Helicone-Session-Id": "sess_abc",
            "session_id": "fallback_sess",
        },
    )
    assert _infer_session_id(req) == "sess_abc"


# ---------------------------------------------------------------------------
# 5. Session inference fallback to session_id
# ---------------------------------------------------------------------------


def test_session_id_fallback_to_lowercase_session_id():
    """Without ``Helicone-Session-Id``, ``session_id`` is the next preference."""
    req = _helicone_request(
        request_id="should_not_be_used",
        properties={"session_id": "fallback_sess"},
    )
    assert _infer_session_id(req) == "fallback_sess"


# ---------------------------------------------------------------------------
# 6. Session inference fallback to request_id (one-call session)
# ---------------------------------------------------------------------------


def test_session_id_fallback_to_request_id():
    """No ``properties`` at all → ``request_id`` becomes the synthetic session id."""
    req = _helicone_request(request_id="req_only", properties={})
    assert _infer_session_id(req) == "req_only"


# ---------------------------------------------------------------------------
# 7. CallRecord shape
# ---------------------------------------------------------------------------


def test_call_record_shape_maps_correctly():
    """CallRecord conversion preserves model, tokens, method, and timestamp."""
    req = _helicone_request(
        request_id="r1",
        provider="anthropic",
        model="claude-sonnet-4-6",
        prompt_tokens=80,
        completion_tokens=12,
        latency_ms=85.5,
        created_at="2026-04-10T11:22:33Z",
    )
    call = _to_call_record(req, session_id="sess_xyz")

    assert call.session_id == "sess_xyz"
    assert call.provider == "anthropic"
    assert call.model == "claude-sonnet-4-6"
    assert call.prompt_tokens == 80
    assert call.completion_tokens == 12
    assert call.latency_ms == 85.5
    # Anthropic models route to ``messages.create``.
    assert call.method == "messages.create"
    assert call.timestamp == datetime(2026, 4, 10, 11, 22, 33, tzinfo=timezone.utc)
    # The Helicone request_id becomes the request_hash for retry-storm dedup.
    assert call.request_hash == "r1"

    # Embedding-shaped model maps to embeddings.create.
    embed_req = _helicone_request(provider="openai", model="text-embedding-3-small")
    embed_req["body"] = {"input": ["hello world"]}
    call_e = _to_call_record(embed_req, session_id="s1")
    assert call_e.method == "embeddings.create"
    assert call_e.raw_request.get("input") == ["hello world"]


def test_infer_method_helpers():
    """Sanity check that the method-routing helper picks the right shape."""
    assert _infer_method(provider="openai", model="text-embedding-3-small") == "embeddings.create"
    assert _infer_method(provider="anthropic", model="claude-haiku-4-5") == "messages.create"
    assert _infer_method(provider="openai", model="gpt-5") == "chat.completions.create"


# ---------------------------------------------------------------------------
# 8. Retroactive replay: tool_loop
# ---------------------------------------------------------------------------


def test_retroactive_replay_fires_tool_loop():
    """Five web_search calls with identical args → tool_loop fires."""
    from token_sentinel.events import CallRecord

    base = datetime(2026, 4, 10, 12, 0, 0, tzinfo=timezone.utc)
    calls: list[CallRecord] = []
    for i in range(5):
        calls.append(
            CallRecord(
                session_id="loop_sess",
                timestamp=base.replace(second=i * 5),
                provider="anthropic",
                model="claude-sonnet-4-6",
                method="messages.create",
                prompt_tokens=120,
                completion_tokens=15,
                latency_ms=80.0,
                request_hash=f"hash_{i}",
                tool_calls=[
                    {
                        "name": "web_search",
                        "arguments": {"query": "tokensentinel best practices"},
                    }
                ],
                user_facing_output=False,
                raw_request={},
                raw_response_meta={},
            )
        )

    from tokensentinel_migrate._retroactive import replay_sessions

    events = replay_sessions(
        {"loop_sess": calls},
        project="test-proj",
    )
    types = [ev.type for ev in events]
    assert "tool_loop" in types
    # Each fired event's ``raised_at`` should match a real call timestamp,
    # not "now" — so backfilled events land at the right point on the
    # dashboard timeline.
    for ev in events:
        assert ev.raised_at >= base
        assert ev.raised_at <= base.replace(second=4 * 5)


# ---------------------------------------------------------------------------
# 9. Retroactive replay: model_misroute
# ---------------------------------------------------------------------------


def test_retroactive_replay_fires_model_misroute():
    """A small classification prompt at gpt-5 → model_misroute."""
    helicone_req = _helicone_request(
        request_id="r1",
        provider="openai",
        model="gpt-5",
        prompt_tokens=40,
        completion_tokens=5,
        prompt="Please classify this sentence as positive or negative.",
    )
    sessions = _group_into_sessions([helicone_req])

    from tokensentinel_migrate._retroactive import replay_sessions

    events = replay_sessions(sessions, project="test-proj")
    types = [ev.type for ev in events]
    assert "model_misroute" in types

    # The fired event's evidence should reflect the routing recommendation.
    fired = next(ev for ev in events if ev.type == "model_misroute")
    assert fired.evidence["model"] == "gpt-5"
    assert fired.evidence["recommended_alternative"] == "gpt-5-mini"


# ---------------------------------------------------------------------------
# 10. Backfill POST: events go to /v1/events:backfill with raised_at
# ---------------------------------------------------------------------------


def test_backfill_post_carries_custom_raised_at():
    """Backfill request body preserves each event's historical raised_at."""
    from token_sentinel.events import LeakEvent

    historical = datetime(2026, 4, 9, 14, 30, 0, tzinfo=timezone.utc)
    events = [
        LeakEvent(
            type="tool_loop",
            confidence=0.85,
            project="test-proj",
            session_id="sess_x",
            rule="v0.tool_loop",
            evidence={"tool": "web_search", "call_count": 4},
            estimated_burn=0.42,
            suggested_action="pause_for_human_review",
            raised_at=historical,
        )
    ]

    captured: dict[str, Any] = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = req.data
        captured["headers"] = dict(req.header_items())
        return _fake_response({"accepted": 1, "rejected": 0})

    with patch("tokensentinel_migrate._backfill.urllib.request.urlopen", fake_urlopen):
        result = _backfill.backfill_events(
            endpoint="https://api.tokensentinel.dev",
            api_key="tsk_test",
            project="test-proj",
            events=events,
        )

    # Endpoint includes the :backfill route + project query param.
    assert "/v1/events:backfill" in captured["url"]
    assert "project=test-proj" in captured["url"]
    body = json.loads(captured["body"])
    assert body["project"] == "test-proj"
    assert len(body["events"]) == 1
    # raised_at survives serialisation as ISO-8601 with the historical date.
    assert body["events"][0]["raised_at"].startswith("2026-04-09T14:30")
    # Auth header is correctly stamped (header keys are title-cased by urllib).
    assert any(
        k.lower() == "authorization" and v == "Bearer tsk_test"
        for k, v in captured["headers"].items()
    )
    assert result == {"accepted": 1, "rejected": 0}


def test_backfill_falls_back_to_v1_events_on_404():
    """Cloud builds without :backfill yet should still accept our traffic."""
    from token_sentinel.events import LeakEvent

    events = [
        LeakEvent(
            type="retry_storm",
            confidence=0.9,
            project="p",
            session_id="s",
            rule="v0.retry_storm",
            evidence={},
            estimated_burn=0.05,
            suggested_action="add_backoff_or_check_upstream_health",
            raised_at=datetime(2026, 4, 9, tzinfo=timezone.utc),
        )
    ]
    urls_seen: list[str] = []

    def fake_urlopen(req, timeout=None):
        urls_seen.append(req.full_url)
        if "/v1/events:backfill" in req.full_url:
            raise HTTPError(
                url=req.full_url,
                code=404,
                msg="not found",
                hdrs=Message(),  # type: ignore[arg-type]
                fp=io.BytesIO(b""),
            )
        return _fake_response({"accepted": 1, "rejected": 0})

    with patch("tokensentinel_migrate._backfill.urllib.request.urlopen", fake_urlopen):
        result = _backfill.backfill_events(
            endpoint="https://api.tokensentinel.dev",
            api_key="tsk_test",
            project="p",
            events=events,
        )

    # First attempt hit :backfill (404), fallback hit /v1/events.
    assert any("/v1/events:backfill" in u for u in urls_seen)
    assert any(u.endswith("/v1/events") for u in urls_seen)
    assert result == {"accepted": 1, "rejected": 0}


# ---------------------------------------------------------------------------
# 11. --dry-run mode does NOT POST
# ---------------------------------------------------------------------------


def test_dry_run_does_not_post_to_cloud():
    """A dry_run migration MUST NOT touch the TokenSentinel cloud.

    The helicone and backfill modules share the global ``urllib.request``
    module, so the test patches ``urllib.request.urlopen`` ONCE with a
    URL-dispatching fake. A backfill URL hitting the fake means the
    importer (incorrectly) tried to POST despite ``dry_run=True``.
    """
    helicone_pages = [
        [
            _helicone_request(
                request_id=f"req_{i}",
                created_at="2026-04-10T12:00:00Z",
                properties={"Helicone-Session-Id": "sess_a"},
            )
            for i in range(3)
        ]
    ]
    page_iter = iter(helicone_pages)
    backfill_calls: list[Any] = []

    def dispatching_urlopen(req, timeout=None):
        url = req.full_url
        if "helicone" in url or "request/query" in url:
            return _fake_response(next(page_iter))
        # Anything that's not the helicone API is the backfill — record it
        # so the test's assertion can verify it stayed silent.
        backfill_calls.append(url)
        return _fake_response({"accepted": 0, "rejected": 0})

    with patch(
        "tokensentinel_migrate.helicone.urllib.request.urlopen",
        dispatching_urlopen,
    ):
        summary = migrate_helicone(
            helicone_api_key="sk-helicone-test",
            tokensentinel_endpoint=None,
            tokensentinel_api_key=None,
            project="test-proj",
            since=None,
            dry_run=True,
        )

    assert summary.dry_run is True
    assert summary.fetched_traces == 3
    # The smoking gun: the backfill module was never invoked.
    assert backfill_calls == []
    # And the summary's backfill counts stay zero.
    assert summary.backfill_accepted == 0
    assert summary.backfill_rejected == 0


# ---------------------------------------------------------------------------
# 12. CLI parses args and routes to importer
# ---------------------------------------------------------------------------


def test_cli_parses_args_and_invokes_importer(capsys):
    """The CLI subcommand should parse arguments and call ``migrate_helicone``."""
    captured_kwargs: dict = {}

    def fake_migrate_helicone(**kwargs):
        captured_kwargs.update(kwargs)
        from tokensentinel_migrate.helicone import MigrationSummary

        s = MigrationSummary(dry_run=kwargs.get("dry_run", False))
        s.fetched_traces = 247
        s.fired_events = 12
        s.events_by_type = {
            "tool_loop": 3,
            "retry_storm": 1,
            "model_misroute": 8,
            "embedding_waste": 0,
        }
        s.burn_by_type = {"tool_loop": 0.83, "retry_storm": 0.21, "model_misroute": 4.42}
        s.estimated_burn_total_usd = 5.46
        return s

    with patch("tokensentinel_migrate.cli.migrate_helicone", fake_migrate_helicone):
        rc = cli_main(
            [
                "helicone",
                "--helicone-api-key",
                "sk-helicone-test",
                "--tokensentinel-endpoint",
                "https://api.tokensentinel.dev",
                "--tokensentinel-api-key",
                "tsk_test",
                "--project",
                "my-agent",
                "--since",
                "2026-04-09",
                "--dry-run",
            ]
        )

    assert rc == 0
    assert captured_kwargs["helicone_api_key"] == "sk-helicone-test"
    assert captured_kwargs["tokensentinel_endpoint"] == "https://api.tokensentinel.dev"
    assert captured_kwargs["tokensentinel_api_key"] == "tsk_test"
    assert captured_kwargs["project"] == "my-agent"
    assert captured_kwargs["dry_run"] is True
    # ``--since 2026-04-09`` parses as midnight UTC.
    since = captured_kwargs["since"]
    assert isinstance(since, datetime)
    assert since == datetime(2026, 4, 9, tzinfo=timezone.utc)

    # CLI rendered the savings summary in the documented format.
    out = capsys.readouterr().out
    assert "tool_loop" in out
    assert "model_misroute" in out
    assert "$5.46" in out
    assert "(dry-run, not posted)" in out


# ---------------------------------------------------------------------------
# Bonus integration check (#13) — full pipeline with mocked HTTP
# ---------------------------------------------------------------------------


def test_full_pipeline_end_to_end_with_mocked_http():
    """Smoke test: fetch -> replay -> backfill, with all HTTP stubbed.

    Not a numbered requirement, but it catches integration regressions
    that the per-module tests miss (e.g., mismatched session-id keys
    between fetch and replay).
    """
    page = [
        _helicone_request(
            request_id=f"r{i}",
            properties={"Helicone-Session-Id": "sess_a"},
            created_at="2026-04-10T12:00:00Z",
            prompt="Classify this as a yes or no",
            provider="openai",
            model="gpt-5",
            prompt_tokens=30,
            completion_tokens=3,
        )
        for i in range(3)
    ]
    pages = iter([page])
    backfill_payloads: list[dict] = []

    def dispatching_urlopen(req, timeout=None):
        url = req.full_url
        if "helicone" in url or "request/query" in url:
            return _fake_response(next(pages))
        # Backfill leg: record + acknowledge.
        body = json.loads(req.data)
        backfill_payloads.append(body)
        return _fake_response({"accepted": len(body.get("events", [])), "rejected": 0})

    with patch(
        "tokensentinel_migrate.helicone.urllib.request.urlopen",
        dispatching_urlopen,
    ):
        summary = migrate_helicone(
            helicone_api_key="sk-helicone-test",
            tokensentinel_endpoint="https://api.tokensentinel.dev",
            tokensentinel_api_key="tsk_test",
            project="proj",
            since=None,
            dry_run=False,
        )

    assert summary.fetched_traces == 3
    assert summary.fired_events >= 1
    assert summary.backfill_accepted >= 1
    # Each posted event preserved its historical timestamp.
    posted_events = [ev for body in backfill_payloads for ev in body["events"]]
    assert all(ev["raised_at"].startswith("2026-04-10T12:00") for ev in posted_events)
