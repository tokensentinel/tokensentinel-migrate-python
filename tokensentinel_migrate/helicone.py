"""Helicone importer.

Helicone went into maintenance mode after Mintlify's March 2026 acquisition
(see ``docs/internal/research/v0_6_cloud_strategy/competitive_paid_features.md``
§Helicone). 16k organisations were on the platform; the most actionable
distribution play for TokenSentinel in 2026 is a frictionless one-command
migrate from Helicone trace data into the TokenSentinel rules + cloud.

This module handles the full pipeline:

1. Paginated fetch from ``POST https://api.helicone.ai/v1/request/query``,
   100 requests per page, with ``Retry-After`` / 429 handling.
2. Session inference: prefer ``properties["Helicone-Session-Id"]``, fall
   back to ``properties["session_id"]``, fall back to ``request_id`` (one-
   call session). Per the founder spec.
3. Conversion of each Helicone request into a :class:`token_sentinel.CallRecord`.
4. Retroactive rule replay via :func:`tokensentinel_migrate._retroactive.replay_sessions`.
5. Backfill POST (when not in dry-run) via :func:`tokensentinel_migrate._backfill.backfill_events`.

The module is stdlib-only (``urllib.request`` for HTTP) so the only
TokenSentinel-side dependency is the SDK itself.

Helicone API quirks to be aware of (see also the README "API quirks" call-out):

- ``request/query`` is a POST, not a GET — pagination + filter both ride in
  the JSON body. ``offset`` and ``limit`` are top-level keys.
- ``created_at`` is ISO-8601 with a ``Z`` suffix (not ``+00:00``); we
  normalise it via ``datetime.fromisoformat`` after stripping the trailing
  ``Z``.
- ``properties`` is a flat string-to-string map. Helicone's own SDK
  recommends ``Helicone-Session-Id`` (canonical mixed case); some
  community SDKs ship ``session_id``. We check both.
- The 429 ``Retry-After`` header is sometimes seconds (an integer string)
  and sometimes an HTTP-date. We support the integer form; HTTP-date
  callers fall through to a default 5s backoff.
- Helicone's older docs reference per-organisation rate limits of ~100 RPM
  on the free tier; paid tiers can burst. The ``Retry-After`` handler is
  still the right design either way.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from token_sentinel.events import CallRecord, LeakEvent

from tokensentinel_migrate._backfill import backfill_events
from tokensentinel_migrate._retroactive import replay_sessions

# Helicone API constants. The endpoint is well-documented; the page size
# matches the canonical example from their docs (100 was the max for the
# request/query endpoint at the time of the Mintlify acquisition).
_HELICONE_QUERY_URL = "https://api.helicone.ai/v1/request/query"
_PAGE_SIZE = 100

# Cap on retry-after waits — a Helicone deploy issue could in principle
# return a multi-hour delay, which we should not honour blindly. 60s is a
# pragmatic ceiling: longer than the typical token-bucket refill, shorter
# than a CLI user would tolerate without canceling.
_MAX_BACKOFF_SECONDS = 60.0

# When 429 arrives without a parseable Retry-After header, default to this.
# Mirrors the typical Helicone rate-limit reset cadence.
_DEFAULT_BACKOFF_SECONDS = 5.0

# Number of consecutive 429s before we give up entirely. The fetch loop is
# expected to recover well before this — six retries with default 5s
# backoff is 30s of wall-clock; longer than that and the API is sick.
_MAX_CONSECUTIVE_RATE_LIMITS = 6


@dataclass
class MigrationSummary:
    """Structured result of a Helicone migration run.

    Returned by :func:`migrate_helicone` and held in memory long enough for
    the CLI to render it. ``events_by_type`` and ``burn_by_type`` are
    parallel dicts keyed on the leak ``type`` field (``tool_loop``,
    ``retry_storm``, etc.) so the summary table at the end of the CLI run
    can render side-by-side counts and dollar values.
    """

    fetched_traces: int = 0
    inferred_sessions: int = 0
    fired_events: int = 0
    events_by_type: dict[str, int] = field(default_factory=dict)
    burn_by_type: dict[str, float] = field(default_factory=dict)
    estimated_burn_total_usd: float = 0.0
    pages_fetched: int = 0
    rate_limit_pauses: int = 0
    backfill_accepted: int = 0
    backfill_rejected: int = 0
    dry_run: bool = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def migrate_helicone(
    *,
    helicone_api_key: str,
    tokensentinel_endpoint: str | None,
    tokensentinel_api_key: str | None,
    project: str,
    since: datetime | None = None,
    dry_run: bool = False,
    rule_config: dict | None = None,
    on_progress: Any = None,
) -> MigrationSummary:
    """Run the Helicone import for ``project``.

    Args:
        helicone_api_key: Customer's Helicone API key. Used as a Bearer
            token on every request to ``api.helicone.ai``.
        tokensentinel_endpoint: TokenSentinel cloud base URL. Required
            unless ``dry_run=True``.
        tokensentinel_api_key: TokenSentinel cloud API key. Required
            unless ``dry_run=True``.
        project: TokenSentinel project name to write the imported events
            into.
        since: Lower bound on Helicone trace ``created_at``. Pages stop
            fetching once any trace older than ``since`` arrives. ``None``
            means "fetch everything Helicone has".
        dry_run: When True, run the full fetch + replay but do NOT POST
            events to TokenSentinel. The returned summary still reports
            counts so the CLI can preview what would be backfilled.
        rule_config: Optional config dict forwarded to :class:`Sentinel`
            during replay. Used to tune rule thresholds (e.g.
            ``{"tool_loop.min_calls": 5}``).
        on_progress: Optional callable accepting (``stage``, ``message``)
            for progress reporting. The CLI passes a print-line wrapper.

    Returns:
        :class:`MigrationSummary` with counts and dollar estimates.
    """
    importer = HeliconeImporter(api_key=helicone_api_key)
    summary = MigrationSummary(dry_run=dry_run)

    def _progress(stage: str, message: str) -> None:
        if on_progress is not None:
            try:
                on_progress(stage, message)
            except Exception:
                # Progress callbacks must never crash the migration. The
                # CLI's wrapper is print-based, but a third-party caller's
                # logger could legitimately raise; swallow.
                pass

    _progress(
        "fetch",
        "Fetching Helicone traces"
        + (f" since {since.date().isoformat()}" if since else "")
        + "...",
    )
    helicone_requests: list[dict[str, Any]] = []
    for page_idx, page in enumerate(importer.iter_pages(since=since), start=1):
        helicone_requests.extend(page)
        summary.pages_fetched = page_idx
        _progress("fetch", f"  page {page_idx}: {len(page)} requests")
        if len(page) < _PAGE_SIZE:
            # Short page = end of dataset (Helicone returns full pages
            # until the last one). Stop fetching to save API quota.
            break
    summary.fetched_traces = len(helicone_requests)
    summary.rate_limit_pauses = importer.rate_limit_pauses

    # Group by inferred session id and convert to CallRecord shape.
    sessions = _group_into_sessions(helicone_requests)
    summary.inferred_sessions = len(sessions)
    _progress(
        "fetch",
        f"Fetched {summary.fetched_traces} traces "
        f"({summary.inferred_sessions} sessions inferred from "
        "heliconeproperty Helicone-Session-Id)",
    )

    # Replay through the SDK rule engine. This is the import's CPU-heavy
    # step; on the order of one rule pass per call across all sessions.
    _progress("replay", "Running TokenSentinel rules retroactively...")
    events = replay_sessions(sessions, project=project, config=rule_config)
    summary.fired_events = len(events)

    # Per-type aggregation for the summary report.
    for ev in events:
        summary.events_by_type[ev.type] = summary.events_by_type.get(ev.type, 0) + 1
        summary.burn_by_type[ev.type] = summary.burn_by_type.get(ev.type, 0.0) + ev.estimated_burn
        summary.estimated_burn_total_usd += ev.estimated_burn

    # Round monetary values to 2 decimals at the boundary so the CLI's
    # output table is stable regardless of the rule's per-call precision.
    summary.estimated_burn_total_usd = round(summary.estimated_burn_total_usd, 2)
    summary.burn_by_type = {k: round(v, 2) for k, v in summary.burn_by_type.items()}

    if dry_run:
        _progress(
            "summary",
            f"{summary.fired_events} leak events would be backfilled (dry-run, not posted)",
        )
        return summary

    if not tokensentinel_endpoint or not tokensentinel_api_key:
        raise ValueError(
            "tokensentinel_endpoint and tokensentinel_api_key are required when dry_run=False"
        )

    _progress(
        "backfill",
        f"Posting {summary.fired_events} events to {tokensentinel_endpoint}...",
    )
    result = backfill_events(
        endpoint=tokensentinel_endpoint,
        api_key=tokensentinel_api_key,
        project=project,
        events=events,
    )
    summary.backfill_accepted = int(result.get("accepted", 0))
    summary.backfill_rejected = int(result.get("rejected", 0))
    _progress(
        "summary",
        f"Backfilled {summary.backfill_accepted} events; "
        f"{summary.backfill_rejected} rejected by cloud.",
    )
    return summary


# ---------------------------------------------------------------------------
# HeliconeImporter — the actual pagination + retry loop
# ---------------------------------------------------------------------------


class HeliconeImporter:
    """Paginated reader for the Helicone request/query endpoint.

    Used internally by :func:`migrate_helicone` but exposed at module level
    for tests and for advanced callers who want to plug a custom replay
    pipeline in front of the cloud-side backfill.
    """

    def __init__(
        self,
        *,
        api_key: str,
        page_size: int = _PAGE_SIZE,
        timeout_seconds: float = 30.0,
    ):
        if not api_key:
            raise ValueError("HeliconeImporter: api_key is required")
        self.api_key = api_key
        self.page_size = page_size
        self.timeout_seconds = timeout_seconds
        # Counter of how many times the loop slept on a 429 before getting
        # a successful response. Surfaced through MigrationSummary so the
        # CLI can warn the user about rate-limit hot spots.
        self.rate_limit_pauses = 0

    def iter_pages(
        self,
        *,
        since: datetime | None = None,
    ) -> Iterator[list[dict[str, Any]]]:
        """Yield successive pages of requests from Helicone, newest-first.

        Stops yielding once a page comes back empty or when any request in
        the latest page is older than ``since`` (the caller provides the
        cutoff; we don't filter at the wire level since Helicone's
        ``request/query`` predicate language is non-trivial — easier to
        truncate post-hoc here).
        """
        offset = 0
        cutoff = _ensure_utc(since) if since is not None else None

        while True:
            page = self._fetch_one_page(offset=offset)
            if not page:
                return  # empty page → end of data

            # Helicone pages are typically sorted newest-first, but defensive
            # filtering on ``cutoff`` is correct regardless of order.
            if cutoff is not None:
                kept = []
                cutoff_breached = False
                for req in page:
                    created_at = _parse_helicone_datetime(req.get("created_at"))
                    if created_at is not None and created_at < cutoff:
                        cutoff_breached = True
                        continue
                    kept.append(req)
                if kept:
                    yield kept
                if cutoff_breached:
                    return  # the rest will all be older — stop fetching
            else:
                yield page

            if len(page) < self.page_size:
                return  # short page = last page
            offset += len(page)

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    def _fetch_one_page(self, *, offset: int) -> list[dict[str, Any]]:
        """POST one page and return the parsed list of requests.

        Handles 429 with ``Retry-After`` by sleeping and retrying; handles
        other 4xx by raising immediately (auth errors, malformed body etc.
        are not transient — surface them so the user can fix their key).
        Network errors are retried up to three times with a 1s/2s/4s
        backoff (mirrors the SDK CloudSink pattern).
        """
        body = json.dumps(
            {
                "filter": "all",
                "offset": offset,
                "limit": self.page_size,
                "sort": {"created_at": "desc"},
            }
        ).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "tokensentinel-migrate/0.1.0",
        }

        rate_limited_count = 0
        network_attempts = 0
        max_network_attempts = 3

        while True:
            req = urllib.request.Request(
                url=_HELICONE_QUERY_URL,
                data=body,
                method="POST",
                headers=headers,
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    raw = resp.read()
                    parsed = json.loads(raw) if raw else []
                    # Helicone's response shape has occasionally been wrapped
                    # in ``{"data": [...]}`` and occasionally been a bare
                    # list. Handle both — the older docs use ``data``, the
                    # newer SDKs sometimes return the bare list.
                    if isinstance(parsed, dict) and "data" in parsed:
                        return list(parsed["data"])
                    if isinstance(parsed, list):
                        return parsed
                    return []
            except urllib.error.HTTPError as exc:
                if exc.code == 429:
                    rate_limited_count += 1
                    if rate_limited_count > _MAX_CONSECUTIVE_RATE_LIMITS:
                        raise RuntimeError(
                            f"tokensentinel-migrate: Helicone rate-limited us "
                            f"{rate_limited_count} times in a row; aborting"
                        ) from exc
                    self.rate_limit_pauses += 1
                    sleep_for = _parse_retry_after(exc)
                    time.sleep(min(sleep_for, _MAX_BACKOFF_SECONDS))
                    continue
                if exc.code in (401, 403):
                    raise RuntimeError(
                        f"tokensentinel-migrate: Helicone returned "
                        f"{exc.code} (check your --helicone-api-key)"
                    ) from exc
                # Other HTTP error — non-retryable. Surface to the user.
                raise RuntimeError(
                    f"tokensentinel-migrate: Helicone returned HTTP {exc.code}: {exc.reason}"
                ) from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                network_attempts += 1
                if network_attempts >= max_network_attempts:
                    raise RuntimeError(
                        f"tokensentinel-migrate: network error fetching Helicone "
                        f"page (offset={offset}) after {max_network_attempts} "
                        f"attempts: {exc!r}"
                    ) from exc
                # Exponential backoff between network retries.
                time.sleep(2 ** (network_attempts - 1))


# ---------------------------------------------------------------------------
# Helicone → CallRecord conversion
# ---------------------------------------------------------------------------


def _group_into_sessions(
    requests: list[dict[str, Any]],
) -> dict[str, list[CallRecord]]:
    """Group Helicone requests by inferred session id and convert each.

    Per the founder spec:
        prefer ``properties["Helicone-Session-Id"]``;
        fall back to ``properties["session_id"]``;
        fall back to ``request_id`` (one-call sessions).
    """
    sessions: dict[str, list[CallRecord]] = {}
    for req in requests:
        session_id = _infer_session_id(req)
        try:
            call = _to_call_record(req, session_id=session_id)
        except Exception:
            # Malformed Helicone trace — skip it. The migration summary
            # still surfaces fetched_traces vs fired_events, so a
            # partial-skip pattern is visible in the totals.
            continue
        sessions.setdefault(session_id, []).append(call)
    return sessions


def _infer_session_id(req: dict[str, Any]) -> str:
    """Return the inferred session id for a Helicone request dict."""
    properties = req.get("properties") or {}
    if not isinstance(properties, dict):
        properties = {}

    # Canonical Helicone session id key. Helicone's docs use the mixed-case
    # form; we keep the comparison case-sensitive because that's what their
    # SDKs ship.
    if properties.get("Helicone-Session-Id"):
        return str(properties["Helicone-Session-Id"])
    if properties.get("session_id"):
        return str(properties["session_id"])

    request_id = req.get("request_id") or req.get("id")
    if request_id:
        return str(request_id)
    # Worst case: synthesize one. Avoid uuid imports — use the timestamp +
    # provider so the id is stable across reruns (idempotent backfill).
    return f"helicone-orphan-{req.get('provider', 'unknown')}-{req.get('created_at', '0')}"


def _to_call_record(req: dict[str, Any], *, session_id: str) -> CallRecord:
    """Convert one Helicone request dict into a TokenSentinel CallRecord.

    Maps:

    - ``provider``    -> ``CallRecord.provider``
    - ``model``       -> ``CallRecord.model``
    - ``prompt_tokens`` / ``completion_tokens`` -> matching fields. Helicone
      sometimes nests these under ``usage`` instead; check both.
    - ``latency_ms``  -> matching field
    - ``created_at``  -> ``CallRecord.timestamp`` (UTC-normalised)
    - ``request_id``  -> ``CallRecord.request_hash`` (Helicone's request_id
      is already an opaque digest; reuse it as the dedup key for retry-storm
      detection — same id == same call retried)
    - ``method``      -> derived from the provider/model:
        * embedding-shaped models (e.g. ``text-embedding-3-small``) are
          mapped to ``embeddings.create`` so the embedding_waste rule sees
          the right ``method`` predicate.
        * everything else is ``chat.completions.create`` (for OpenAI-shaped
          providers) or ``messages.create`` (for Anthropic).
    - ``raw_request.messages`` -> from Helicone's ``prompt`` field. We
      structure it as a single user message so model_misroute's
      keyword-on-content scan works without rewriting the rule. Helicone
      stores prompts as either a string or a JSON-encoded message array;
      both forms are normalised here.
    """
    provider = str(req.get("provider", "unknown")).lower()
    model = str(req.get("model", "unknown"))
    usage = req.get("usage") or {}
    prompt_tokens = int(
        req.get("prompt_tokens") or usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    )
    completion_tokens = int(
        req.get("completion_tokens")
        or usage.get("completion_tokens")
        or usage.get("output_tokens")
        or 0
    )
    latency_ms = float(req.get("latency_ms") or req.get("latency") or 0.0)
    created_at = _parse_helicone_datetime(req.get("created_at")) or datetime.now(timezone.utc)
    request_hash = str(req.get("request_id") or req.get("id") or session_id)
    method = _infer_method(provider=provider, model=model)
    raw_request = _build_raw_request(req)

    return CallRecord(
        session_id=session_id,
        timestamp=created_at,
        provider=provider,
        model=model,
        method=method,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        request_hash=request_hash,
        tool_calls=[],
        user_facing_output=False,
        raw_request=raw_request,
        raw_response_meta={},
    )


def _infer_method(*, provider: str, model: str) -> str:
    """Best-effort mapping from (provider, model) to TokenSentinel method.

    The :class:`EmbeddingWasteRule` predicate is
    ``c.method.endswith("embeddings.create")``; if we mis-classify an
    embedding call as a chat completion the rule never fires and the
    customer's wasted spend goes invisible. Conservative heuristic: any
    model name containing "embedding" maps to embeddings.create.
    """
    lowered = model.lower()
    if "embedding" in lowered or "embed" in lowered:
        return "embeddings.create"
    if provider == "anthropic":
        return "messages.create"
    # OpenAI, OpenAI-compatible, generic — chat.completions.create is the
    # canonical method name for the rule engine's session walks.
    return "chat.completions.create"


def _build_raw_request(req: dict[str, Any]) -> dict[str, Any]:
    """Construct a ``raw_request`` shape that the rule engine can read.

    The model_misroute rule scans ``raw_request["messages"]`` for
    classification keywords; the embedding_waste rule reads
    ``raw_request["input"]`` to detect duplicate embeds. We populate both
    from Helicone's ``prompt`` field, normalising the various shapes
    Helicone has shipped over time.
    """
    raw_request: dict[str, Any] = {}
    prompt = req.get("prompt")
    if prompt is None:
        prompt = req.get("body", {}).get("prompt") if isinstance(req.get("body"), dict) else None
    body = req.get("body") if isinstance(req.get("body"), dict) else {}

    # If Helicone shipped a structured ``messages`` list (newer SDKs),
    # forward it untouched — that's the most informative shape for the
    # rules.
    messages = body.get("messages") if isinstance(body, dict) else None
    if isinstance(messages, list):
        raw_request["messages"] = messages
    elif isinstance(prompt, list):
        raw_request["messages"] = prompt
    elif isinstance(prompt, str):
        raw_request["messages"] = [{"role": "user", "content": prompt}]
    else:
        raw_request["messages"] = []

    # Embedding ``input`` is shipped under ``body.input`` for OpenAI-shaped
    # embedding calls. Forward when present so embedding_waste's hashing
    # has something to dedup against.
    if isinstance(body, dict) and "input" in body:
        raw_request["input"] = body["input"]
    elif "input" in req:
        raw_request["input"] = req["input"]

    return raw_request


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_helicone_datetime(value: Any) -> datetime | None:
    """Parse Helicone's ISO-8601 timestamps into a UTC ``datetime``.

    Helicone has shipped timestamps as ``2026-04-09T12:34:56Z``,
    ``2026-04-09T12:34:56.789Z``, and ``2026-04-09T12:34:56+00:00`` over
    the years. ``datetime.fromisoformat`` on Python 3.10+ handles the last
    form natively; the trailing-Z forms need a one-character substitution
    before the parse.

    Returns ``None`` on unparseable input rather than raising — a malformed
    timestamp on one trace shouldn't take down the whole migration.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return _ensure_utc(value)
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return _ensure_utc(dt)


def _ensure_utc(dt: datetime) -> datetime:
    """Ensure ``dt`` carries UTC tzinfo — naive datetimes are treated as UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_retry_after(exc: urllib.error.HTTPError) -> float:
    """Extract seconds-to-sleep from a 429 response's Retry-After header.

    Returns ``_DEFAULT_BACKOFF_SECONDS`` when the header is missing or in
    the (unsupported) HTTP-date form.
    """
    headers = getattr(exc, "headers", None)
    if headers is None:
        return _DEFAULT_BACKOFF_SECONDS
    raw = None
    try:
        raw = headers.get("Retry-After")
    except Exception:
        return _DEFAULT_BACKOFF_SECONDS
    if raw is None:
        return _DEFAULT_BACKOFF_SECONDS
    try:
        return max(0.5, float(raw))
    except (TypeError, ValueError):
        # HTTP-date form. We could parse via ``email.utils.parsedate_to_datetime``
        # but the canonical Helicone behaviour is integer seconds; default
        # is good enough.
        return _DEFAULT_BACKOFF_SECONDS


# Re-export LeakEvent so callers don't have to dual-import from
# token_sentinel — keeps the migrate module's public surface self-contained
# for the most common scripted use case.
__all__ = [
    "HeliconeImporter",
    "MigrationSummary",
    "migrate_helicone",
    "LeakEvent",
]
