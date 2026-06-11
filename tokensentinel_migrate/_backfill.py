"""Chunked POST to TokenSentinel cloud's events-backfill endpoint.

Wire format mirrors :file:`cloud/DESIGN.md`'s ``POST /v1/events`` exactly —
``project`` + ``events[]`` JSON body, ``Authorization: Bearer <api_key>``,
each event carries its own ``raised_at`` timestamp. The importer needs the
custom ``raised_at`` because each backfilled event represents a leak that
*would have* fired at the original Helicone trace's ``created_at`` time, not
at "now" — without it the dashboard's "tokens saved this week" counter would
attribute every imported event to today.

Endpoint resolution:

The spec calls for ``POST /v1/events:backfill?project=X``. As a transition
measure (parallel cloud-side agents are extending the V0.5 endpoint), this
module first tries ``/v1/events:backfill``; on a 404 it falls back to the
existing ``/v1/events`` endpoint, which already accepts a custom
``raised_at`` per event (verified in
``cloud/backend/tokensentinel_cloud/ingest.py``). The fall-through means a
v0.5 cloud build accepts our backfill traffic today; a v0.6 cloud build with
the dedicated ``:backfill`` route gets routed there transparently. No
behaviour change is needed in the importer when the cloud lights up the new
route.

TODO(cloud-team): once ``POST /v1/events:backfill?project=X`` is live in the
cloud, the v0.5 fall-through can be deleted and ``_FALLBACK_TO_V1_EVENTS``
flipped to ``False`` (or removed). Tracked in the parent agent's queue.
"""

from __future__ import annotations

import dataclasses
import json
import time
import urllib.error
import urllib.request
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from token_sentinel.events import LeakEvent


# Default chunk size matches the cloud's V0.5 ingest model (max 1000 events
# per request, but the design docs use 50-event batches as the steady-state
# norm — keeps each POST cheap to retry on transient network failures).
_DEFAULT_CHUNK_SIZE = 50

# Retries between attempts. Same shape as the SDK's CloudSink — three tries
# with exponential backoff (1s, 2s, 4s).
_RETRY_BACKOFFS_SECONDS = (1.0, 2.0, 4.0)

# When ``True`` (the default for v0.1.0), a 404 on ``/v1/events:backfill``
# falls back to ``/v1/events`` so a v0.5 cloud build still accepts our
# traffic. Flip to ``False`` once the cloud's :backfill route is live
# everywhere we ship to.
_FALLBACK_TO_V1_EVENTS = True


def _event_to_wire(event: LeakEvent, sdk_version: str) -> dict[str, Any]:
    """Serialise a :class:`LeakEvent` into the on-the-wire dict.

    Mirrors :func:`token_sentinel.cloud_client._event_to_wire` so the
    backfill payload looks indistinguishable from a live ingestion payload
    from the cloud's perspective. ``raised_at`` is converted from datetime
    to ISO-8601 string; everything else is a straight ``dataclasses.asdict``
    walk.
    """
    payload: dict[str, Any] = dataclasses.asdict(event)
    raised_at = payload.get("raised_at")
    if isinstance(raised_at, datetime):
        payload["raised_at"] = raised_at.isoformat()
    payload["sdk_version"] = sdk_version
    return payload


def backfill_events(
    *,
    endpoint: str,
    api_key: str,
    project: str,
    events: list[LeakEvent],
    sdk_version: str = "0.5.0",
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    timeout_seconds: float = 30.0,
) -> dict[str, int]:
    """POST ``events`` to the backfill endpoint in chunks of ``chunk_size``.

    Returns a dict with ``accepted`` and ``rejected`` counts summed over all
    chunks. On network failure the chunk is retried up to three times
    (exponential backoff). Persistent failure raises ``RuntimeError`` so the
    CLI surfaces it to the user — backfill is an explicit one-shot
    operation, NOT the SDK's fire-and-forget hot path.

    Args:
        endpoint: TokenSentinel cloud base URL, e.g. ``https://api.tokensentinel.dev``.
        api_key: A ``tsk_…`` API key with write access to ``project``.
        project: The TokenSentinel project name to write to.
        events: List of :class:`LeakEvent` instances. Each ``raised_at`` is
            preserved on the wire so the dashboard timeline reflects the
            historical Helicone timestamps, not "now".
        sdk_version: Stamped onto every event for the cloud's `sdk_version`
            column. Defaults to a recent SDK version since we use the SDK to
            generate the events.
        chunk_size: How many events per POST. The cloud's per-request cap is
            1000; 50 is the conservative default (cheaper retries on flaky
            networks, smaller failure blast radius).
        timeout_seconds: Per-request socket timeout.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if not events:
        return {"accepted": 0, "rejected": 0}

    base = endpoint.rstrip("/")
    accepted = 0
    rejected = 0

    for start in range(0, len(events), chunk_size):
        chunk = events[start : start + chunk_size]
        body = json.dumps(
            {
                "project": project,
                "events": [_event_to_wire(ev, sdk_version) for ev in chunk],
            },
            default=str,
        ).encode("utf-8")
        result = _post_with_retry(
            base=base,
            project=project,
            api_key=api_key,
            sdk_version=sdk_version,
            body=body,
            timeout_seconds=timeout_seconds,
        )
        accepted += int(result.get("accepted", 0))
        rejected += int(result.get("rejected", 0))

    return {"accepted": accepted, "rejected": rejected}


def _post_with_retry(
    *,
    base: str,
    project: str,
    api_key: str,
    sdk_version: str,
    body: bytes,
    timeout_seconds: float,
) -> dict[str, Any]:
    """POST one chunk. Tries ``:backfill`` first, falls back to ``/v1/events``.

    Implements the route-discovery dance described in the module docstring:
    the new endpoint may not exist on the v0.5 cloud build, so we treat a
    404 there as "not deployed yet" and retry against the V0.5 endpoint.
    Once the cloud team flips the deploy switch, the first attempt always
    wins and the fallback path is dead code (which we'll then delete).

    Real network errors (timeouts, 500s, transient connection drops) are
    retried up to ``len(_RETRY_BACKOFFS_SECONDS)`` times via exponential
    backoff. After exhaustion we raise ``RuntimeError`` with the last
    underlying exception chained.
    """
    url_backfill = f"{base}/v1/events:backfill?project={project}"
    url_v1 = f"{base}/v1/events"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": f"tokensentinel-migrate/0.1.0 (sdk {sdk_version})",
    }

    last_exc: BaseException | None = None
    backfill_unavailable = False
    for attempt in range(len(_RETRY_BACKOFFS_SECONDS)):
        # Pick the URL for this attempt. If we already discovered :backfill
        # is 404 on this endpoint, don't keep retrying it — go straight to
        # the v0.5 fallback.
        if backfill_unavailable or not _FALLBACK_TO_V1_EVENTS:
            url = url_backfill if not backfill_unavailable else url_v1
        else:
            url = url_backfill
        req = urllib.request.Request(url=url, data=body, method="POST", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                status = getattr(resp, "status", None)
                if status is None:
                    getter = getattr(resp, "getcode", None)
                    status = getter() if callable(getter) else 200
                if not (200 <= int(status) < 300):
                    # Non-2xx response without an HTTPError — older urllib
                    # quirk. Convert to an HTTPError so the retry path
                    # treats it uniformly with the urllib.error.HTTPError
                    # branch below.
                    raise urllib.error.HTTPError(
                        url, int(status), f"unexpected status {status}", resp.headers, None
                    )
                # Try to parse the response body for accepted/rejected counts.
                try:
                    raw = resp.read()
                    parsed = json.loads(raw) if raw else {}
                except Exception:
                    parsed = {}
                return parsed if isinstance(parsed, dict) else {}
        except urllib.error.HTTPError as exc:
            # 404 on :backfill triggers the fallback to /v1/events. Set the
            # flag and retry immediately (no backoff — same logical attempt,
            # just a different URL).
            if (
                _FALLBACK_TO_V1_EVENTS
                and exc.code == 404
                and url == url_backfill
                and not backfill_unavailable
            ):
                backfill_unavailable = True
                continue
            last_exc = exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_exc = exc

        # Sleep before the next attempt. Skip the sleep on the final attempt.
        if attempt < len(_RETRY_BACKOFFS_SECONDS) - 1:
            time.sleep(_RETRY_BACKOFFS_SECONDS[attempt])

    raise RuntimeError(
        f"tokensentinel-migrate: failed to POST chunk after "
        f"{len(_RETRY_BACKOFFS_SECONDS)} attempts; last error: {last_exc!r}"
    ) from last_exc
