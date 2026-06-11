"""Langfuse Cloud importer.

Langfuse is the largest OSS LLM observability project; Langfuse Cloud is
the hosted SaaS offering of the same stack. Per the V0.6.1 founder
Decision 3, Langfuse is one of the two remaining top-tier migration
targets after the Helicone importer shipped in v0.1.0 — see also
``docs/internal/research/v0_6_cloud_strategy/competitive_paid_features.md``
for the broader migration thesis.

This module mirrors :mod:`tokensentinel_migrate.helicone` exactly: it
exposes a single :func:`import_runs` entry point that fetches traces from
Langfuse Cloud, normalises each ``GENERATION`` observation into a
:class:`token_sentinel.CallRecord`, replays the resulting sessions through
the TokenSentinel rule engine via the shared
:func:`tokensentinel_migrate._retroactive.replay_sessions`, and (unless
``dry_run=True``) backfills the leak events to TokenSentinel cloud via
:func:`tokensentinel_migrate._backfill.backfill_events`.

The Langfuse Cloud API documented at https://api.reference.langfuse.com:

- ``GET /api/public/traces``: paginated list of traces. Query params:
  ``fromTimestamp``, ``toTimestamp``, ``limit`` (max 100), ``page``. The
  response shape is ``{"data": [...], "meta": {"totalItems", "totalPages",
  "page", "limit"}}``.
- Each trace carries an ``observations`` array; only ``type == "GENERATION"``
  observations represent real LLM calls (others are spans / events / tool
  invocations the user explicitly tagged). The importer drops non-GENERATION
  observations early.
- Auth is HTTP Basic with ``public_key`` as username and ``secret_key`` as
  password. The two-key model is documented and stable.

Langfuse-specific quirks documented through the codebase here:

- Self-hosted users point at their own deployment URL; we expose
  ``--langfuse-base-url`` for that (the CLI's default is
  ``https://cloud.langfuse.com``).
- ``observations[].usage`` may be ``{"unit": "TOKENS"}`` or
  ``{"unit": "CHARACTERS"}`` — we only honour TOKENS. CHARACTERS units fall
  back to ``prompt_tokens=0/completion_tokens=0`` so the CallRecord still
  exists and gets replayed through e.g. retry_storm / tool_loop rules; the
  dollar estimate just under-counts those particular calls.
- A trace may have zero GENERATION observations (a pure ``SPAN`` / tool
  trace). We drop those traces from the output entirely.
- ``startTime`` / ``endTime`` are ISO-8601 with either ``Z`` or ``+00:00``
  suffixes; we normalise both via the same helper as the Helicone module.
- Rate-limit handling mirrors Helicone: 429 → ``Retry-After`` integer
  seconds or HTTP-date → sleep up to 60s, retry up to six times before
  raising.
"""

from __future__ import annotations

import base64
import hashlib
import json
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

# Default Langfuse Cloud base URL. Self-hosted callers override via the
# ``base_url`` kwarg / ``--langfuse-base-url`` CLI flag.
_LANGFUSE_DEFAULT_BASE_URL = "https://cloud.langfuse.com"

# Per the Langfuse Cloud docs (api.reference.langfuse.com) the
# ``/traces`` endpoint caps ``limit`` at 100; the default is 50. We use
# the max to minimise round-trips on bulk imports.
_PAGE_SIZE = 100

# Mirror Helicone's backoff caps so the two importers behave consistently
# from a CLI-user point of view. See ``helicone.py`` for rationale.
_MAX_BACKOFF_SECONDS = 60.0
_DEFAULT_BACKOFF_SECONDS = 5.0
_MAX_CONSECUTIVE_RATE_LIMITS = 6


@dataclass
class MigrationSummary:
    """Structured result of a Langfuse migration run.

    Same shape as :class:`tokensentinel_migrate.helicone.MigrationSummary`
    so the CLI's renderer is provider-agnostic — adding new importers
    later only requires another row in the subcommand registry, not a new
    summary type.
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


def migrate_langfuse(
    *,
    public_key: str,
    secret_key: str,
    tokensentinel_endpoint: str | None,
    tokensentinel_api_key: str | None,
    project: str,
    since: datetime | None = None,
    until: datetime | None = None,
    base_url: str = _LANGFUSE_DEFAULT_BASE_URL,
    dry_run: bool = False,
    rule_config: dict[str, Any] | None = None,
    on_progress: Callable[[str, str], None] | None = None,
) -> MigrationSummary:
    """Run the Langfuse import for ``project``.

    Args:
        public_key: Langfuse public key (``pk-lf-...``). Username half of
            the HTTP Basic credential.
        secret_key: Langfuse secret key (``sk-lf-...``). Password half of
            the HTTP Basic credential.
        tokensentinel_endpoint: TokenSentinel cloud base URL. Required
            unless ``dry_run=True``.
        tokensentinel_api_key: TokenSentinel cloud API key. Required
            unless ``dry_run=True``.
        project: TokenSentinel project name to write the imported events
            into.
        since: Lower bound on Langfuse trace ``timestamp``. Translated to
            the ``fromTimestamp`` query param. ``None`` means "fetch
            everything Langfuse has".
        until: Upper bound on Langfuse trace ``timestamp``. Translated to
            the ``toTimestamp`` query param. ``None`` means "up to now".
        base_url: Langfuse Cloud / self-hosted base URL. Defaults to
            ``https://cloud.langfuse.com``; self-hosted callers point at
            their own deployment.
        dry_run: When True, run the full fetch + replay but do NOT POST
            events to TokenSentinel.
        rule_config: Optional config dict forwarded to :class:`Sentinel`
            during replay.
        on_progress: Optional callable accepting (``stage``, ``message``)
            for progress reporting.

    Returns:
        :class:`MigrationSummary` with counts and dollar estimates.
    """
    importer = LangfuseImporter(
        public_key=public_key,
        secret_key=secret_key,
        base_url=base_url,
    )
    summary = MigrationSummary(dry_run=dry_run)

    def _progress(stage: str, message: str) -> None:
        if on_progress is not None:
            try:
                on_progress(stage, message)
            except Exception:
                # Progress callbacks must never crash the migration; see
                # the rationale on the parallel branch in helicone.py.
                pass

    _progress(
        "fetch",
        "Fetching Langfuse traces"
        + (f" since {since.date().isoformat()}" if since else "")
        + "...",
    )
    traces: list[dict[str, Any]] = []
    for page_idx, page in enumerate(importer.iter_pages(since=since, until=until), start=1):
        traces.extend(page)
        summary.pages_fetched = page_idx
        _progress("fetch", f"  page {page_idx}: {len(page)} traces")
    summary.fetched_traces = len(traces)
    summary.rate_limit_pauses = importer.rate_limit_pauses

    sessions = _group_into_sessions(traces)
    summary.inferred_sessions = len(sessions)
    _progress(
        "fetch",
        f"Fetched {summary.fetched_traces} traces "
        f"({summary.inferred_sessions} sessions inferred from "
        "trace.sessionId)",
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


# Alias matching the shared importer-contract name in the spec. The two
# names co-exist so existing scripts that import ``migrate_langfuse`` keep
# working and new scripts can use the canonical ``import_runs`` name.
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
    base_url: str = _LANGFUSE_DEFAULT_BASE_URL,
    rule_config: dict[str, Any] | None = None,
) -> MigrationSummary:
    """Provider-agnostic entry-point alias for :func:`migrate_langfuse`.

    The ``source_api_key`` is the combined ``"<public_key>|<secret_key>"``
    pair, matching the spec's "Langfuse: 'pk-...|sk-...' (combined)"
    contract — handy when callers want one importer-agnostic kwarg.
    """
    public_key, _, secret_key = source_api_key.partition("|")
    if not public_key or not secret_key:
        raise ValueError(
            "Langfuse import_runs: source_api_key must be of the form "
            "'<public_key>|<secret_key>' (got a single-token key)"
        )
    return migrate_langfuse(
        public_key=public_key,
        secret_key=secret_key,
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
# LangfuseImporter — the actual pagination + retry loop
# ---------------------------------------------------------------------------


class LangfuseImporter:
    """Paginated reader for the Langfuse Cloud ``/traces`` endpoint."""

    def __init__(
        self,
        *,
        public_key: str,
        secret_key: str,
        base_url: str = _LANGFUSE_DEFAULT_BASE_URL,
        page_size: int = _PAGE_SIZE,
        timeout_seconds: float = 30.0,
    ):
        if not public_key:
            raise ValueError("LangfuseImporter: public_key is required")
        if not secret_key:
            raise ValueError("LangfuseImporter: secret_key is required")
        self.public_key = public_key
        self.secret_key = secret_key
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
        """Yield successive pages of traces from Langfuse.

        Stops yielding once the server reports ``page >= totalPages`` or
        a short page comes back (defensive: matches Helicone's stop
        condition for callers that bypass the meta block).
        """
        page = 1
        cutoff_since = _ensure_utc(since) if since is not None else None
        cutoff_until = _ensure_utc(until) if until is not None else None
        while True:
            raw_page, meta = self._fetch_one_page(page=page, since=cutoff_since, until=cutoff_until)
            if not raw_page:
                return
            yield raw_page
            total_pages = int(meta.get("totalPages", 0)) if isinstance(meta, dict) else 0
            if total_pages and page >= total_pages:
                return
            if len(raw_page) < self.page_size:
                return
            page += 1

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    def _auth_header(self) -> str:
        """HTTP Basic header value composed from ``public_key:secret_key``."""
        token = base64.b64encode(f"{self.public_key}:{self.secret_key}".encode()).decode("ascii")
        return f"Basic {token}"

    def _build_url(
        self,
        *,
        page: int,
        since: datetime | None,
        until: datetime | None,
    ) -> str:
        """Construct the paginated ``/api/public/traces`` URL."""
        params = [f"page={page}", f"limit={self.page_size}"]
        if since is not None:
            params.append(f"fromTimestamp={_to_iso_z(since)}")
        if until is not None:
            params.append(f"toTimestamp={_to_iso_z(until)}")
        return f"{self.base_url}/api/public/traces?" + "&".join(params)

    def _fetch_one_page(
        self,
        *,
        page: int,
        since: datetime | None,
        until: datetime | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """GET one page; return (data, meta) tuple.

        Same 429 / Retry-After dance as :class:`HeliconeImporter`: sleep up
        to 60s, retry up to six consecutive 429s before raising. Other 4xx
        are non-retryable (auth errors etc.); 5xx and network errors are
        retried with exponential backoff up to three attempts.
        """
        url = self._build_url(page=page, since=since, until=until)
        headers = {
            "Authorization": self._auth_header(),
            "Accept": "application/json",
            "User-Agent": "tokensentinel-migrate/0.1.0",
        }

        rate_limited_count = 0
        network_attempts = 0
        max_network_attempts = 3

        while True:
            req = urllib.request.Request(url=url, method="GET", headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    raw = resp.read()
                    parsed = json.loads(raw) if raw else {}
                    if not isinstance(parsed, dict):
                        return [], {}
                    data = parsed.get("data") or []
                    meta = parsed.get("meta") or {}
                    if not isinstance(data, list):
                        data = []
                    if not isinstance(meta, dict):
                        meta = {}
                    return list(data), dict(meta)
            except urllib.error.HTTPError as exc:
                if exc.code == 429:
                    rate_limited_count += 1
                    if rate_limited_count > _MAX_CONSECUTIVE_RATE_LIMITS:
                        raise RuntimeError(
                            f"tokensentinel-migrate: Langfuse rate-limited us "
                            f"{rate_limited_count} times in a row; aborting"
                        ) from exc
                    self.rate_limit_pauses += 1
                    sleep_for = _parse_retry_after(exc)
                    time.sleep(min(sleep_for, _MAX_BACKOFF_SECONDS))
                    continue
                if exc.code in (401, 403):
                    raise RuntimeError(
                        f"tokensentinel-migrate: Langfuse returned "
                        f"{exc.code} (check your --langfuse-public-key / "
                        f"--langfuse-secret-key)"
                    ) from exc
                raise RuntimeError(
                    f"tokensentinel-migrate: Langfuse returned HTTP {exc.code}: {exc.reason}"
                ) from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                network_attempts += 1
                if network_attempts >= max_network_attempts:
                    raise RuntimeError(
                        f"tokensentinel-migrate: network error fetching Langfuse "
                        f"page (page={page}) after {max_network_attempts} "
                        f"attempts: {exc!r}"
                    ) from exc
                time.sleep(2 ** (network_attempts - 1))


# ---------------------------------------------------------------------------
# Langfuse → CallRecord conversion
# ---------------------------------------------------------------------------


def _group_into_sessions(
    traces: list[dict[str, Any]],
) -> dict[str, list[CallRecord]]:
    """Group Langfuse trace observations by inferred session id.

    The session-id heuristic per the founder spec:
        prefer ``trace.sessionId``;
        fall back to ``trace.id`` (single-call trace).

    Only ``GENERATION`` observations become CallRecords; ``SPAN`` and
    ``EVENT`` rows are skipped (they don't represent real LLM calls and
    the rule engine doesn't have a meaningful interpretation for them).
    """
    sessions: dict[str, list[CallRecord]] = {}
    for trace in traces:
        if not isinstance(trace, dict):
            continue
        session_id = _infer_session_id(trace)
        observations = trace.get("observations") or []
        if not isinstance(observations, list):
            continue
        for obs in observations:
            if not isinstance(obs, dict):
                continue
            if str(obs.get("type", "")).upper() != "GENERATION":
                continue
            try:
                call = _to_call_record(obs, trace=trace, session_id=session_id)
            except Exception:
                # Malformed observation — skip. Mirrors the helicone
                # importer's partial-skip pattern.
                continue
            sessions.setdefault(session_id, []).append(call)
    return sessions


def _infer_session_id(trace: dict[str, Any]) -> str:
    """Return the inferred session id for a Langfuse trace dict."""
    session_id = trace.get("sessionId")
    if session_id:
        return str(session_id)
    trace_id = trace.get("id")
    if trace_id:
        return str(trace_id)
    # Synthetic fallback for the rare case where Langfuse ships a trace
    # without an id. Stable across reruns so the backfill stays idempotent.
    return f"langfuse-orphan-{trace.get('timestamp', '0')}"


def _to_call_record(
    obs: dict[str, Any],
    *,
    trace: dict[str, Any],
    session_id: str,
) -> CallRecord:
    """Convert one Langfuse GENERATION observation into a CallRecord.

    Maps:
      - ``obs.model``     -> ``CallRecord.model``
      - provider          -> inferred from model name (see :func:`_infer_provider`)
      - ``obs.usage.input`` / ``obs.usage.output`` -> token counts (only
        when ``unit == "TOKENS"``; CHARACTERS units leave the counts at 0)
      - ``endTime - startTime`` -> ``latency_ms``
      - ``obs.startTime`` -> ``timestamp``
      - method            -> ``"messages.create"`` by default (Langfuse
        doesn't preserve the underlying SDK method name; pick the safer
        default that doesn't activate :class:`EmbeddingWasteRule` falsely)
      - tool_calls        -> parsed from ``obs.output`` if it looks like
        Anthropic's tool_use shape or OpenAI's function_call shape
      - request_hash      -> SHA-256 of (session_id + obs.id), truncated
        to 16 hex chars. Stable across reruns -> idempotent backfill.
    """
    model = str(obs.get("model") or "unknown")
    provider = _infer_provider(model=model, obs=obs)
    prompt_tokens, completion_tokens = _extract_usage(obs)
    start_time = _parse_langfuse_datetime(obs.get("startTime")) or datetime.now(timezone.utc)
    end_time = _parse_langfuse_datetime(obs.get("endTime"))
    latency_ms = 0.0
    if end_time is not None:
        latency_ms = max(0.0, (end_time - start_time).total_seconds() * 1000.0)

    request_hash = _compute_request_hash(session_id=session_id, obs_id=obs.get("id"))
    tool_calls = _extract_tool_calls(obs.get("output"))

    return CallRecord(
        session_id=session_id,
        timestamp=start_time,
        provider=provider,
        model=model,
        # Langfuse doesn't preserve the SDK method; ``messages.create`` is
        # the conservative default. Embedding calls land here too but
        # without a way to detect them from Langfuse's shape we accept the
        # embedding_waste rule's slight under-firing on Langfuse imports
        # (documented in the README's "Langfuse gotchas" section).
        method="messages.create",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        request_hash=request_hash,
        tool_calls=tool_calls,
        # Langfuse doesn't track the "user-facing output" flag the way the
        # SDK does. Default conservatively to True so the tool_loop rule's
        # ``user_facing_output`` predicate (which suppresses spurious
        # firings on visible UI streams) errs on the side of suppression
        # rather than over-firing. Per spec.
        user_facing_output=True,
        raw_request=_build_raw_request(obs, trace=trace),
        raw_response_meta={},
    )


def _infer_provider(*, model: str, obs: dict[str, Any]) -> str:
    """Best-effort provider inference from observation ``model`` name."""
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
    # Some Langfuse customers stash a ``modelParameters.provider`` field
    # via the SDK; honour it if present even though the API spec doesn't
    # require it. Wrapping in a defensive isinstance keeps the call safe
    # against arbitrarily-shaped customer extensions.
    params = obs.get("modelParameters")
    if isinstance(params, dict):
        explicit = params.get("provider")
        if isinstance(explicit, str) and explicit:
            return explicit.lower()
    return "unknown"


def _extract_usage(obs: dict[str, Any]) -> tuple[int, int]:
    """Return ``(prompt_tokens, completion_tokens)`` from observation usage.

    Returns ``(0, 0)`` when the usage block is missing, malformed, or
    quoted in CHARACTERS instead of TOKENS — those CallRecords still
    propagate through replay so non-token rules (tool_loop, retry_storm)
    fire correctly, the dollar estimate just under-counts.
    """
    usage = obs.get("usage")
    if not isinstance(usage, dict):
        return (0, 0)
    unit = str(usage.get("unit", "TOKENS")).upper()
    if unit and unit != "TOKENS":
        # Non-TOKENS units (CHARACTERS, IMAGES, ...) — can't safely
        # interpret as tokens.
        return (0, 0)
    try:
        prompt = int(usage.get("input") or 0)
    except (TypeError, ValueError):
        prompt = 0
    try:
        completion = int(usage.get("output") or 0)
    except (TypeError, ValueError):
        completion = 0
    return (prompt, completion)


def _extract_tool_calls(output: Any) -> list[dict[str, Any]]:
    """Best-effort parse of tool/function calls from observation output.

    Handles both Anthropic's ``tool_use`` blocks (``content`` list with
    ``type=="tool_use"`` items) and OpenAI's ``function_call`` /
    ``tool_calls`` shape. Returns an empty list when no recognisable tool
    invocations are found — the tool_loop rule treats absence as "no tool
    activity" which is the safe default.
    """
    if output is None:
        return []
    # Anthropic shape: content list with tool_use blocks.
    if isinstance(output, dict):
        content = output.get("content")
        if isinstance(content, list):
            tools = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    tools.append(
                        {
                            "name": str(block.get("name", "")),
                            "arguments": block.get("input") or {},
                        }
                    )
            if tools:
                return tools
        # OpenAI shape: ``tool_calls`` array (newer) or single
        # ``function_call`` object (legacy).
        oai_tools = output.get("tool_calls")
        if isinstance(oai_tools, list):
            parsed: list[dict[str, Any]] = []
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
        fn = output.get("function_call")
        if isinstance(fn, dict):
            return [
                {
                    "name": str(fn.get("name", "")),
                    "arguments": fn.get("arguments") or {},
                }
            ]
    return []


def _build_raw_request(
    obs: dict[str, Any],
    *,
    trace: dict[str, Any],
) -> dict[str, Any]:
    """Construct a minimal ``raw_request`` shape from the observation.

    Per the founder spec: Langfuse sometimes strips the raw input so the
    default is ``{}``. When ``obs.input`` is present and looks like a
    messages array, we forward it so the model_misroute rule's
    keyword-on-content scan works. ``trace.input`` is the secondary
    source — useful for traces whose generation observation didn't store
    a top-level prompt.
    """
    raw_request: dict[str, Any] = {}
    obs_input = obs.get("input")
    if isinstance(obs_input, list):
        raw_request["messages"] = obs_input
    elif isinstance(obs_input, dict):
        messages = obs_input.get("messages")
        if isinstance(messages, list):
            raw_request["messages"] = messages
        # Embedding-style ``input`` field — forward verbatim for the
        # embedding_waste rule's dedup hashing.
        if "input" in obs_input:
            raw_request["input"] = obs_input["input"]
    elif isinstance(obs_input, str) and obs_input:
        raw_request["messages"] = [{"role": "user", "content": obs_input}]

    if "messages" not in raw_request:
        trace_input = trace.get("input")
        if isinstance(trace_input, list):
            raw_request["messages"] = trace_input
        elif isinstance(trace_input, str) and trace_input:
            raw_request["messages"] = [{"role": "user", "content": trace_input}]

    return raw_request


def _compute_request_hash(*, session_id: str, obs_id: Any) -> str:
    """SHA-256(session_id + obs_id), hex-truncated to 16 chars."""
    payload = f"{session_id}|{obs_id or ''}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_langfuse_datetime(value: Any) -> datetime | None:
    """Parse Langfuse ISO-8601 timestamps into a UTC ``datetime``.

    Handles the trailing-``Z`` and ``+00:00`` forms. Returns ``None`` on
    unparseable input rather than raising — a malformed timestamp on one
    observation shouldn't take down the whole migration.
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
    """Ensure ``dt`` carries UTC tzinfo."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _to_iso_z(dt: datetime) -> str:
    """Format ``dt`` as ``YYYY-MM-DDTHH:MM:SSZ`` for Langfuse query params."""
    return _ensure_utc(dt).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_retry_after(exc: urllib.error.HTTPError) -> float:
    """Extract seconds-to-sleep from a 429 ``Retry-After`` header.

    Supports both the integer-seconds form (``Retry-After: 12``) and the
    HTTP-date form (``Retry-After: Wed, 21 Oct 2026 07:28:00 GMT``). The
    Langfuse docs don't pin a specific form so we honour both — see also
    the rationale on the Helicone side, which defaulted to integer-only.
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
    # Integer-seconds form first — cheapest to parse.
    try:
        return max(0.5, float(raw))
    except (TypeError, ValueError):
        pass
    # HTTP-date form. ``parsedate_to_datetime`` returns an aware datetime
    # when the input includes a timezone; we clamp the resulting delta to
    # [0.5, _MAX_BACKOFF_SECONDS] so a misformatted date can't trigger a
    # multi-day sleep.
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
    "LangfuseImporter",
    "LeakEvent",
    "MigrationSummary",
    "import_runs",
    "migrate_langfuse",
]
