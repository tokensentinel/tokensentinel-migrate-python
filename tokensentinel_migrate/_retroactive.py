"""Replay imported traces through the TokenSentinel SDK rule engine.

The importer fetches traces from a third-party tool, normalises each one
into a :class:`token_sentinel.CallRecord`, then walks the resulting
sessions through a freshly-constructed :class:`token_sentinel.Sentinel` in
``mode='log'``. The Sentinel's leak handler appends each fired event to a
list, which the caller batches and POSTs to the cloud's backfill endpoint.

Two correctness properties matter here:

1. **No live cloud chatter.** The Sentinel we construct does NOT pass
   ``cloud_endpoint`` or ``api_key``, so its CloudSink and PolicyClient
   subsystems never spawn — the rule engine runs entirely in-process. The
   importer's :mod:`_backfill` module is the only thing that talks to the
   cloud, and it does so AFTER the replay completes with the full event
   batch in hand.

2. **Original timestamps are preserved.** Each :class:`LeakEvent` produced
   by the SDK's rule engine has a ``raised_at`` set to ``datetime.now(UTC)``
   by default. We OVERWRITE that with the timestamp of the call that
   triggered the rule firing — using the timestamp of the call whose
   ``record_call`` produced the event. Without this every backfilled event
   would attribute to "today" and the dashboard's tokens-saved-this-week
   counter would incorrectly count migrated history as current activity.

The session ordering is enforced by sorting calls within each session by
their ``timestamp`` field before replay. The rules' window-based logic (the
60-second ``tool_loop`` window, the 30-second ``retry_storm`` window) only
makes sense if calls arrive in chronological order, mirroring the SDK's
production hot path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from token_sentinel.events import CallRecord, LeakEvent


def replay_sessions(
    sessions: dict[str, list[CallRecord]],
    *,
    project: str,
    config: dict | None = None,
) -> list[LeakEvent]:
    """Replay grouped sessions through a fresh Sentinel; return fired events.

    Args:
        sessions: Dict mapping ``session_id`` to a list of CallRecords. The
            list is sorted by ``timestamp`` before replay.
        project: Project name to stamp on every emitted LeakEvent.
        config: Optional rule-config dict passed straight to ``Sentinel``.
            Defaults to an empty dict, i.e. all rules use their defaults.

    Returns:
        List of :class:`LeakEvent` instances, each with ``raised_at`` set to
        the timestamp of the call that triggered the rule firing.

    Implementation note: we re-instantiate the Sentinel per session rather
    than reusing a single one across sessions. The Sentinel's de-duplication
    state is bounded per-session, but bouncing between many session ids in
    a long migration walks across the LRU cap and risks losing the dedup
    record for early sessions before they finish replaying. Per-session
    isolation also matches the V0.6 policy plane's per-session burn budget
    semantics — exactly what we want when "replaying as if these had been
    live."
    """
    out: list[LeakEvent] = []

    for _session_id, calls in sessions.items():
        if not calls:
            continue
        sorted_calls = sorted(calls, key=lambda c: c.timestamp)
        events_for_session = _replay_one_session(
            sorted_calls,
            project=project,
            config=config,
        )
        out.extend(events_for_session)

    return out


def _replay_one_session(
    sorted_calls: list[CallRecord],
    *,
    project: str,
    config: dict | None,
) -> list[LeakEvent]:
    """Replay one session's calls; return the LeakEvents fired, time-stamped.

    Factored out of :func:`replay_sessions` so the leak-handler closure
    binds to a fresh, per-call ``events_for_session`` list rather than a
    loop-scoped variable (which ruff B023 correctly flags as a footgun
    even though the closures here run synchronously inside the loop body).
    """
    from token_sentinel import Sentinel

    cfg = dict(config) if config else {}

    # Build a fresh Sentinel per session. ``cloud_endpoint`` and
    # ``api_key`` are intentionally not passed — keeps CloudSink /
    # PolicyClient threads from ever spawning. The replay is offline.
    # ``policy_endpoint=None`` is explicit defensive belt-and-braces in
    # case the default ``_POLICY_DEFAULT`` ever changes to a
    # non-cloud-derived value.
    sentinel = Sentinel(
        project=project,
        mode="log",
        config=cfg,
        policy_endpoint=None,
    )

    # Per-session collector. The handler closure captures this list and
    # records every fired event for this session. We track which events
    # have already had their raised_at re-stamped via a set of object ids
    # so the per-call sweep below is idempotent.
    events_for_session: list[LeakEvent] = []
    already_stamped: set[int] = set()

    @sentinel.on_leak
    def _collector(event: LeakEvent) -> None:
        events_for_session.append(event)

    for call in sorted_calls:
        try:
            sentinel.record_call(call)
        except Exception:
            # The rule engine is exception-safe per-rule, but if a future
            # bug propagates we don't want one bad call to abort the whole
            # migration. Continue past the bad call.
            pass

        # Re-stamp every newly-fired event with this call's timestamp.
        # That's the moment the rule "would have" fired in production. We
        # track stamped events by ``id()`` so re-entry into this loop
        # never double-stamps an event with a later timestamp.
        for ev in events_for_session:
            if id(ev) in already_stamped:
                continue
            ev.raised_at = call.timestamp
            already_stamped.add(id(ev))

    # Best-effort shutdown of the per-session Sentinel. Even with no cloud
    # sink there's nothing to flush, but ``.close()`` is the documented
    # teardown — calling it keeps us forward-compatible if a future
    # Sentinel version spawns optional background work.
    try:
        sentinel.close(timeout=1.0)
    except Exception:
        pass

    return events_for_session
