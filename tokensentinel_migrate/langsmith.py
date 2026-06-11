"""LangSmith importer.

LangSmith is LangChain's hosted observability product — the default
trace destination for any LangChain / LangGraph agent. Per the V0.6.1
founder Decision 3, LangSmith is the second of the two remaining top-tier
migration targets after the Helicone importer shipped in v0.1.0. See
also ``docs/internal/research/v0_6_cloud_strategy/competitive_paid_features.md``
for the broader migration thesis.

The module mirrors :mod:`tokensentinel_migrate.helicone` and
:mod:`tokensentinel_migrate.langfuse` exactly: a single :func:`import_runs`
entry point fetches runs from LangSmith, normalises each ``run_type=="llm"``
run into a :class:`token_sentinel.CallRecord`, replays the resulting
sessions through the shared rule engine, and (unless ``dry_run=True``)
backfills the leak events to TokenSentinel cloud.

LangSmith's data API (https://docs.smith.langchain.com/reference/data_format):

- ``POST /runs/query``: paginated list of runs. Body fields:
  ``start_time``, ``end_time``, ``limit`` (default 100, max 100),
  ``cursor`` (opaque pagination token), ``execution_order=1`` to keep
  the result set to top-level runs only.
- Auth is ``X-API-Key: ls__...`` — the same key shape LangSmith's
  Python SDK ships.
- Each run has a ``run_type`` field; only ``"llm"`` runs represent real
  LLM API calls. ``chain`` / ``tool`` / ``retriever`` runs are dropped
  (the rule engine doesn't have a meaningful interpretation for them on
  their own; the tool_loop rule reads ``CallRecord.tool_calls`` instead,
  which we populate from the LLM run's structured output).

LangSmith-specific quirks documented through the codebase here:

- Token counts live on the run at the top level for newer LangSmith
  versions (``run.prompt_tokens`` / ``run.completion_tokens``); older
  versions stash them under ``run.extra.invocation_params.usage``. The
  importer checks both, in that order.
- Provider inference is heuristic: ``run.extra.invocation_params._type``
  carries strings like ``"anthropic-chat"`` / ``"openai-chat"`` /
  ``"chat-google-vertexai"``; we regex on the prefix and fall back to
  inferring from the model name (same heuristic as the Langfuse importer).
- ``start_time`` / ``end_time`` are ISO-8601 with optional fractional
  seconds and ``Z`` suffix; we normalise both via the same helper as the
  other importers.
- Rate-limit handling mirrors the other importers: 429 → ``Retry-After``
  integer seconds or HTTP-date → sleep up to 60s, retry up to six
  consecutive 429s before raising.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

from token_sentinel.events import CallRecord, LeakEvent

from tokensentinel_migrate._backfill import backfill_events
from tokensentinel_migrate._retroactive import replay_sessions

# Default LangSmith Cloud base URL. Self-hosted callers can override via
# the ``base_url`` kwarg; LangSmith's enterprise SKU runs at a separate
# tenant URL per customer so this is parameterised but rarely overridden
# in practice.
_LANGSMITH_DEFAULT_BASE_URL = "https://api.smith.langchain.com"

# Per the docs (docs.smith.langchain.com/reference/data_format) the
# ``/runs/query`` endpoint caps ``limit`` at 100.
_PAGE_SIZE = 100

# Mirror Helicone / Langfuse backoff so CLI behaviour is consistent.
_MAX_BACKOFF_SECONDS = 60.0
_DEFAULT_BACKOFF_SECONDS = 5.0
_MAX_CONSECUTIVE_RATE_LIMITS = 6

# Provider inference from LangSmith's ``invocation_params._type`` strings.
# Common values observed in the wild:
#   anthropic-chat / chat-anthropic   -> anthropic
#   openai-chat / chat-openai          -> openai
#   chat-google-vertexai / google-genai -> google
#   chat-mistralai                     -> mistral
_TYPE_TO_PROVIDER = re.compile(
    r"^(chat-)?(?P<vendor>anthropic|openai|google|vertex|mistral|cohere|meta|together)"
)


@dataclass
class MigrationSummary:
    """Structured result of a LangSmith migration run.

    Same shape as the helicone / langfuse summaries — the CLI's renderer
    is intentionally provider-agnostic so adding new importers is just a
    subcommand registration.
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


def migrate_langsmith(
    *,
    api_key: str,
    tokensentinel_endpoint: str | None,
    tokensentinel_api_key: str | None,
    project: str,
    since: datetime | None = None,
    until: datetime | None = None,
    base_url: str = _LANGSMITH_DEFAULT_BASE_URL,
    dry_run: bool = False,
    rule_config: dict[str, Any] | None = None,
    on_progress: Callable[[str, str], None] | None = None,
) -> MigrationSummary:
    """Run the LangSmith import for ``project``.

    Args:
        api_key: LangSmith API key (``ls__...``). Sent as ``X-API-Key``.
        tokensentinel_endpoint: TokenSentinel cloud base URL. Required
            unless ``dry_run=True``.
        tokensentinel_api_key: TokenSentinel cloud API key. Required
            unless ``dry_run=True``.
        project: TokenSentinel project name to write the imported events
            into.
        since: Lower bound on run ``start_time``. Translated to
            ``start_time`` in the query body.
        until: Upper bound on run ``start_time``. Translated to
            ``end_time`` in the query body.
        base_url: LangSmith API base URL. Defaults to the public Cloud.
        dry_run: When True, run the full fetch + replay but do NOT POST
            events to TokenSentinel.
        rule_config: Optional config dict forwarded to :class:`Sentinel`.
        on_progress: Optional callable accepting (``stage``, ``message``)
            for progress reporting.

    Returns:
        :class:`MigrationSummary` with counts and dollar estimates.
    """
    importer = LangSmithImporter(api_key=api_key, base_url=base_url)
    summary = MigrationSummary(dry_run=dry_run)

    def _progress(stage: str, message: str) -> None:
        if on_progress is not None:
            try:
                on_progress(stage, message)
            except Exception:
                # Mirrors the helicone / langfuse rationale.
                pass

    _progress(
        "fetch",
        "Fetching LangSmith runs" + (f" since {since.date().isoformat()}" if since else "") + "...",
    )
    runs: list[dict[str, Any]] = []
    for page_idx, page in enumerate(importer.iter_pages(since=since, until=until), start=1):
        runs.extend(page)
        summary.pages_fetched = page_idx
        _progress("fetch", f"  page {page_idx}: {len(page)} runs")
    summary.fetched_traces = len(runs)
    summary.rate_limit_pauses = importer.rate_limit_pauses

    sessions = _group_into_sessions(runs)
    summary.inferred_sessions = len(sessions)
    _progress(
        "fetch",
        f"Fetched {summary.fetched_traces} runs "
        f"({summary.inferred_sessions} sessions inferred from "
        "run.session_id)",
    )

    _progress("replay", "Running TokenSentinel rules retroactively...")
    events = replay_sessions(sessions, project=project, config=rule_config)
    summary.fired_events = len(events)

    for ev in events:
        summary.events_by_type[ev.type] = summary.events_by_type.get(ev.type, 0) + 1
        summary.burn_by_type[ev.type] = summary.burn_by_type.get(ev.type, 0.0) + ev.estimated_burn
        summary.estimated_burn_total_usd += ev.estimated_burn

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


def import_runs(
    *,
    source_api_key: str,
    tokensentinel_endpoint: str | None,
    tokensentinel_api_key: str | None,
    project: str,
    since: datetime | None = None,
    until: datetime | None = None,
    dry_run: bool = False,
    on_progress: Callable[[str, str], None] | None = None,
    base_url: str = _LANGSMITH_DEFAULT_BASE_URL,
    rule_config: dict[str, Any] | None = None,
) -> MigrationSummary:
    """Provider-agnostic entry-point alias for :func:`migrate_langsmith`.

    Implements the spec's shared importer contract: a single
    ``source_api_key`` kwarg that maps directly onto the LangSmith
    ``X-API-Key`` header.
    """
    return migrate_langsmith(
        api_key=source_api_key,
        tokensentinel_endpoint=tokensentinel_endpoint,
        tokensentinel_api_key=tokensentinel_api_key,
        project=project,
        since=since,
        until=until,
        base_url=base_url,
        dry_run=dry_run,
        rule_config=rule_config,
        on_progress=on_progress,
    )


# ---------------------------------------------------------------------------
# LangSmithImporter — pagination + retry loop
# ---------------------------------------------------------------------------


class LangSmithImporter:
    """Paginated reader for the LangSmith ``/runs/query`` endpoint.

    Pagination uses an opaque ``cursor`` field rather than offset/page-
    number; we mirror the cursor from the previous response into the next
    request's body until the server hands back a ``cursors.next == null``
    response.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = _LANGSMITH_DEFAULT_BASE_URL,
        page_size: int = _PAGE_SIZE,
        timeout_seconds: float = 30.0,
    ):
        if not api_key:
            raise ValueError("LangSmithImporter: api_key is required")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.page_size = page_size
        self.timeout_seconds = timeout_seconds
        self.rate_limit_pauses = 0

    def iter_pages(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> Iterator[list[dict[str, Any]]]:
        """Yield successive pages of runs from LangSmith.

        Stops yielding when the server hands back ``cursors.next == null``
        (or empty / missing) or when an empty page comes back.
        """
        cursor: str | None = None
        cutoff_since = _ensure_utc(since) if since is not None else None
        cutoff_until = _ensure_utc(until) if until is not None else None
        while True:
            raw_runs, next_cursor = self._fetch_one_page(
                cursor=cursor, since=cutoff_since, until=cutoff_until
            )
            if not raw_runs:
                return
            yield raw_runs
            if not next_cursor:
                return
            cursor = next_cursor

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    def _build_body(
        self,
        *,
        cursor: str | None,
        since: datetime | None,
        until: datetime | None,
    ) -> bytes:
        """Compose the JSON body for one ``POST /runs/query``."""
        body: dict[str, Any] = {
            "limit": self.page_size,
            "execution_order": 1,  # top-level runs only
        }
        if cursor is not None:
            body["cursor"] = cursor
        if since is not None:
            body["start_time"] = _to_iso_z(since)
        if until is not None:
            body["end_time"] = _to_iso_z(until)
        return json.dumps(body).encode("utf-8")

    def _fetch_one_page(
        self,
        *,
        cursor: str | None,
        since: datetime | None,
        until: datetime | None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """POST one page; return (runs, next_cursor)."""
        url = f"{self.base_url}/runs/query"
        body = self._build_body(cursor=cursor, since=since, until=until)
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "tokensentinel-migrate/0.1.0",
        }

        rate_limited_count = 0
        network_attempts = 0
        max_network_attempts = 3

        while True:
            req = urllib.request.Request(url=url, data=body, method="POST", headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    raw = resp.read()
                    parsed = json.loads(raw) if raw else {}
                    if not isinstance(parsed, dict):
                        return [], None
                    runs = parsed.get("runs") or []
                    if not isinstance(runs, list):
                        runs = []
                    cursors = parsed.get("cursors") or {}
                    next_cursor: str | None = None
                    if isinstance(cursors, dict):
                        raw_next = cursors.get("next")
                        if isinstance(raw_next, str) and raw_next:
                            next_cursor = raw_next
                    return list(runs), next_cursor
            except urllib.error.HTTPError as exc:
                if exc.code == 429:
                    rate_limited_count += 1
                    if rate_limited_count > _MAX_CONSECUTIVE_RATE_LIMITS:
                        raise RuntimeError(
                            f"tokensentinel-migrate: LangSmith rate-limited us "
                            f"{rate_limited_count} times in a row; aborting"
                        ) from exc
                    self.rate_limit_pauses += 1
                    sleep_for = _parse_retry_after(exc)
                    time.sleep(min(sleep_for, _MAX_BACKOFF_SECONDS))
                    continue
                if exc.code in (401, 403):
                    raise RuntimeError(
                        f"tokensentinel-migrate: LangSmith returned "
                        f"{exc.code} (check your --langsmith-api-key)"
                    ) from exc
                raise RuntimeError(
                    f"tokensentinel-migrate: LangSmith returned HTTP {exc.code}: {exc.reason}"
                ) from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                network_attempts += 1
                if network_attempts >= max_network_attempts:
                    raise RuntimeError(
                        f"tokensentinel-migrate: network error fetching LangSmith "
                        f"page after {max_network_attempts} attempts: {exc!r}"
                    ) from exc
                time.sleep(2 ** (network_attempts - 1))


# ---------------------------------------------------------------------------
# LangSmith → CallRecord conversion
# ---------------------------------------------------------------------------


def _group_into_sessions(
    runs: list[dict[str, Any]],
) -> dict[str, list[CallRecord]]:
    """Group LangSmith runs by inferred session id.

    Per the spec: prefer ``run.session_id``, fall back to ``run.id``.
    Only ``run_type=="llm"`` runs become CallRecords.
    """
    sessions: dict[str, list[CallRecord]] = {}
    for run in runs:
        if not isinstance(run, dict):
            continue
        if str(run.get("run_type", "")).lower() != "llm":
            continue
        session_id = _infer_session_id(run)
        try:
            call = _to_call_record(run, session_id=session_id)
        except Exception:
            continue
        sessions.setdefault(session_id, []).append(call)
    return sessions


def _infer_session_id(run: dict[str, Any]) -> str:
    """Return the inferred session id for a LangSmith run dict."""
    session_id = run.get("session_id")
    if session_id:
        return str(session_id)
    run_id = run.get("id")
    if run_id:
        return str(run_id)
    return f"langsmith-orphan-{run.get('start_time', '0')}"


def _to_call_record(run: dict[str, Any], *, session_id: str) -> CallRecord:
    """Convert one LangSmith ``llm``-type run into a CallRecord.

    Maps:
      - ``run.extra.invocation_params.model`` -> ``model``
      - provider          -> regex-inferred from ``invocation_params._type``,
        falling back to model-name heuristic
      - tokens            -> top-level ``prompt_tokens`` / ``completion_tokens``
        first, then ``extra.invocation_params.usage`` for older formats
      - ``end_time - start_time`` -> ``latency_ms``
      - ``run.start_time`` -> ``timestamp``
      - method            -> ``"messages.create"`` default
      - tool_calls        -> parsed from ``run.outputs`` if recognisable
      - request_hash      -> SHA-256 of (session_id + run.id), truncated
    """
    extra = run.get("extra")
    if not isinstance(extra, dict):
        extra = {}
    invocation_params = extra.get("invocation_params")
    if not isinstance(invocation_params, dict):
        invocation_params = {}

    model = str(invocation_params.get("model") or run.get("name") or "unknown")
    provider = _infer_provider(invocation_params=invocation_params, model=model)
    prompt_tokens, completion_tokens = _extract_usage(run=run, invocation_params=invocation_params)

    start_time = _parse_langsmith_datetime(run.get("start_time")) or datetime.now(timezone.utc)
    end_time = _parse_langsmith_datetime(run.get("end_time"))
    latency_ms = 0.0
    if end_time is not None:
        latency_ms = max(0.0, (end_time - start_time).total_seconds() * 1000.0)

    request_hash = _compute_request_hash(session_id=session_id, run_id=run.get("id"))
    tool_calls = _extract_tool_calls(run.get("outputs"))

    return CallRecord(
        session_id=session_id,
        timestamp=start_time,
        provider=provider,
        model=model,
        method="messages.create",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        request_hash=request_hash,
        tool_calls=tool_calls,
        user_facing_output=True,
        raw_request=_build_raw_request(run),
        raw_response_meta={},
    )


def _infer_provider(
    *,
    invocation_params: dict[str, Any],
    model: str,
) -> str:
    """Best-effort provider inference.

    Tries the LangChain ``_type`` string first (most reliable when
    present), then the model name (mirrors the Langfuse heuristic so the
    two importers behave consistently for the same underlying model).
    """
    raw_type = invocation_params.get("_type")
    if isinstance(raw_type, str):
        match = _TYPE_TO_PROVIDER.match(raw_type.lower())
        if match:
            vendor = match.group("vendor")
            if vendor == "vertex":
                return "google"
            return vendor
    # Model-name fallback.
    lowered = model.lower()
    if "claude" in lowered:
        return "anthropic"
    if "gpt" in lowered or "o1" in lowered or "o3" in lowered or "o4" in lowered:
        return "openai"
    if "gemini" in lowered:
        return "google"
    if "mistral" in lowered or "mixtral" in lowered:
        return "mistral"
    if "llama" in lowered:
        return "meta"
    return "unknown"


def _extract_usage(
    *,
    run: dict[str, Any],
    invocation_params: dict[str, Any],
) -> tuple[int, int]:
    """Return ``(prompt_tokens, completion_tokens)`` for a LangSmith run.

    Checks top-level fields first (newer LangSmith format), then falls
    back to ``invocation_params.usage`` (older format). Returns (0, 0)
    when nothing parseable is present — the CallRecord still propagates
    through replay for non-token rules.
    """
    # Newer LangSmith ships these as top-level integers.
    prompt_top = run.get("prompt_tokens")
    completion_top = run.get("completion_tokens")
    if prompt_top is not None or completion_top is not None:
        try:
            prompt = int(prompt_top or 0)
        except (TypeError, ValueError):
            prompt = 0
        try:
            completion = int(completion_top or 0)
        except (TypeError, ValueError):
            completion = 0
        return (prompt, completion)
    # Older format: nested under invocation_params.
    usage = invocation_params.get("usage")
    if isinstance(usage, dict):
        try:
            prompt = int(
                usage.get("prompt_tokens") or usage.get("input_tokens") or usage.get("input") or 0
            )
        except (TypeError, ValueError):
            prompt = 0
        try:
            completion = int(
                usage.get("completion_tokens")
                or usage.get("output_tokens")
                or usage.get("output")
                or 0
            )
        except (TypeError, ValueError):
            completion = 0
        return (prompt, completion)
    return (0, 0)


def _extract_tool_calls(outputs: Any) -> list[dict[str, Any]]:
    """Best-effort parse of tool calls from a run's outputs.

    LangChain serialises outputs as either ``{"generations": [[...]]}``
    (chat model output) or ``{"output": ...}`` (raw LLM output). Inside,
    tool calls show up either as Anthropic ``tool_use`` content blocks
    or as OpenAI ``tool_calls`` / ``function_call`` shapes. We walk both.
    """
    if not isinstance(outputs, dict):
        return []
    # Chat model output: generations[0][0].message.tool_calls.
    generations = outputs.get("generations")
    if isinstance(generations, list) and generations:
        first = generations[0]
        if isinstance(first, list) and first:
            inner = first[0]
            if isinstance(inner, dict):
                message = inner.get("message") or inner
                if isinstance(message, dict):
                    tc = message.get("tool_calls")
                    if isinstance(tc, list):
                        parsed: list[dict[str, Any]] = []
                        for entry in tc:
                            if not isinstance(entry, dict):
                                continue
                            parsed.append(
                                {
                                    "name": str(entry.get("name", "")),
                                    "arguments": entry.get("args") or entry.get("arguments") or {},
                                }
                            )
                        if parsed:
                            return parsed
                    # Anthropic tool_use inside content list.
                    content = message.get("content")
                    if isinstance(content, list):
                        a_parsed: list[dict[str, Any]] = []
                        for block in content:
                            if not isinstance(block, dict):
                                continue
                            if block.get("type") == "tool_use":
                                a_parsed.append(
                                    {
                                        "name": str(block.get("name", "")),
                                        "arguments": block.get("input") or {},
                                    }
                                )
                        if a_parsed:
                            return a_parsed
    # Generic OpenAI shape directly under outputs.
    oai_tools = outputs.get("tool_calls")
    if isinstance(oai_tools, list):
        parsed = []
        for tc in oai_tools:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            if not isinstance(fn, dict):
                fn = {}
            parsed.append(
                {
                    "name": str(fn.get("name", "")),
                    "arguments": fn.get("arguments") or {},
                }
            )
        if parsed:
            return parsed
    return []


def _build_raw_request(run: dict[str, Any]) -> dict[str, Any]:
    """Best-effort raw_request shape from a LangSmith run."""
    raw_request: dict[str, Any] = {}
    inputs = run.get("inputs")
    if isinstance(inputs, dict):
        messages = inputs.get("messages")
        if isinstance(messages, list):
            raw_request["messages"] = messages
        elif "input" in inputs:
            # Single-string prompt or embedding input.
            value = inputs["input"]
            if isinstance(value, str):
                raw_request["messages"] = [{"role": "user", "content": value}]
            else:
                raw_request["input"] = value
    return raw_request


def _compute_request_hash(*, session_id: str, run_id: Any) -> str:
    """SHA-256(session_id + run_id), hex-truncated to 16 chars."""
    payload = f"{session_id}|{run_id or ''}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_langsmith_datetime(value: Any) -> datetime | None:
    """Parse LangSmith ISO-8601 timestamps into a UTC ``datetime``."""
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
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _to_iso_z(dt: datetime) -> str:
    return _ensure_utc(dt).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_retry_after(exc: urllib.error.HTTPError) -> float:
    """Mirror of the Langfuse helper; supports integer + HTTP-date."""
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
        pass
    try:
        dt = parsedate_to_datetime(raw)
        if dt is None:
            return _DEFAULT_BACKOFF_SECONDS
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = (dt - datetime.now(timezone.utc)).total_seconds()
        if delta <= 0:
            return _DEFAULT_BACKOFF_SECONDS
        return max(0.5, min(delta, _MAX_BACKOFF_SECONDS))
    except (TypeError, ValueError):
        return _DEFAULT_BACKOFF_SECONDS


__all__ = [
    "LangSmithImporter",
    "LeakEvent",
    "MigrationSummary",
    "import_runs",
    "migrate_langsmith",
]
