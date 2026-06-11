"""CLI entry point — ``python -m tokensentinel_migrate <provider>``.

The CLI is intentionally thin; all real work lives in the per-provider
modules. The CLI's job is:

1. Parse arguments and validate them.
2. Construct progress callback.
3. Invoke the matching ``migrate_<provider>`` function.
4. Render the human-readable summary table at the end.

The summary block is the customer-visible artifact — tightly formatted to
match the example in the founder spec so the screenshot users post on
social media reads like a clean migration report rather than a debug dump.

All three current importers (Helicone, Langfuse, LangSmith) return
:class:`MigrationSummary` dataclasses with identical fields, so the
``_render_summary`` helper is provider-agnostic. The per-provider runner
functions handle each importer's distinct auth surface; the renderer just
duck-types over the shared summary shape.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from typing import Any

from tokensentinel_migrate.helicone import migrate_helicone
from tokensentinel_migrate.langfuse import migrate_langfuse
from tokensentinel_migrate.langsmith import migrate_langsmith


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns process exit code (0 = success)."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.provider == "helicone":
        return _run_helicone(args)
    if args.provider == "langfuse":
        return _run_langfuse(args)
    if args.provider == "langsmith":
        return _run_langsmith(args)

    parser.error(f"Unknown provider: {args.provider}")
    return 2  # parser.error exits, but mypy doesn't know that


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tokensentinel-migrate",
        description=(
            "Import LLM observability traces from Helicone, Langfuse, or "
            "LangSmith into TokenSentinel cloud. Replays the imported "
            "traces through TokenSentinel's leak rules and backfills the "
            "resulting events so the dashboard reflects the savings you "
            "would have realised had TokenSentinel been wired in."
        ),
    )
    sub = parser.add_subparsers(dest="provider", required=True)

    # ------------------------------ helicone ------------------------------
    helicone_p = sub.add_parser("helicone", help="Import from Helicone (api.helicone.ai)")
    helicone_p.add_argument(
        "--helicone-api-key",
        required=True,
        help="Your Helicone API key (read access is sufficient).",
    )
    _add_shared_flags(helicone_p)

    # ------------------------------ langfuse ------------------------------
    # Langfuse uses HTTP Basic with a (public_key, secret_key) pair; the
    # two keys are independent, neither alone is enough. We surface both
    # as required flags so the importer never has to validate auth shape.
    langfuse_p = sub.add_parser(
        "langfuse", help="Import from Langfuse Cloud or self-hosted Langfuse"
    )
    langfuse_p.add_argument(
        "--langfuse-public-key",
        required=True,
        help="Your Langfuse public key (pk-lf-...).",
    )
    langfuse_p.add_argument(
        "--langfuse-secret-key",
        required=True,
        help="Your Langfuse secret key (sk-lf-...).",
    )
    langfuse_p.add_argument(
        "--langfuse-base-url",
        default="https://cloud.langfuse.com",
        help=(
            "Base URL for Langfuse. Defaults to the public Cloud. "
            "Self-hosted callers point at their own deployment."
        ),
    )
    _add_shared_flags(langfuse_p)

    # ------------------------------ langsmith -----------------------------
    langsmith_p = sub.add_parser(
        "langsmith", help="Import from LangSmith (api.smith.langchain.com)"
    )
    langsmith_p.add_argument(
        "--langsmith-api-key",
        required=True,
        help="Your LangSmith API key (ls__...). Sent as X-API-Key.",
    )
    langsmith_p.add_argument(
        "--langsmith-base-url",
        default="https://api.smith.langchain.com",
        help=("Base URL for the LangSmith API. Defaults to the public Cloud."),
    )
    _add_shared_flags(langsmith_p)

    return parser


def _add_shared_flags(sub_p: argparse.ArgumentParser) -> None:
    """Attach the flags that every provider subcommand carries.

    Kept as a helper rather than inlined three times so the help-text wording
    stays in sync across providers and adding a new provider in v0.2 is just
    one subcommand and one call to this helper.
    """
    sub_p.add_argument(
        "--tokensentinel-endpoint",
        default=None,
        help=(
            "TokenSentinel cloud base URL, e.g. https://api.tokensentinel.dev. "
            "Required unless --dry-run is set."
        ),
    )
    sub_p.add_argument(
        "--tokensentinel-api-key",
        default=None,
        help=("TokenSentinel cloud API key (tsk_…). Required unless --dry-run."),
    )
    sub_p.add_argument(
        "--project",
        required=True,
        help="The TokenSentinel project name to write events into.",
    )
    sub_p.add_argument(
        "--since",
        default=None,
        help=(
            "ISO-8601 date or datetime (e.g. 2026-04-09 or "
            "2026-04-09T00:00:00Z). Only traces newer than this are "
            "imported. If omitted, fetches everything the source has."
        ),
    )
    sub_p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run the full fetch + replay pipeline but do NOT post events "
            "to TokenSentinel. Use to preview the migration."
        ),
    )


def _make_progress() -> Any:
    """Build the print-line progress callback used by every runner."""

    def progress(stage: str, message: str) -> None:
        # The "[migrate] " prefix matches the founder spec's example output;
        # keep the exact text stable so docs / blog posts can quote it.
        sys.stdout.write(f"[migrate] {message}\n")
        sys.stdout.flush()

    return progress


def _check_cloud_creds(args: argparse.Namespace) -> int | None:
    """Validate cloud-creds invariant shared by every runner.

    Returns the exit code to bubble up on failure, or ``None`` when the
    args are consistent. Centralised so the three runners stay one-liner
    thin and the failure mode is identical across providers.
    """
    if not args.dry_run and (not args.tokensentinel_endpoint or not args.tokensentinel_api_key):
        sys.stderr.write(
            "[migrate] error: --tokensentinel-endpoint and "
            "--tokensentinel-api-key are required without --dry-run.\n"
        )
        return 2
    return None


def _run_helicone(args: argparse.Namespace) -> int:
    rc = _check_cloud_creds(args)
    if rc is not None:
        return rc
    since = _parse_since(args.since) if args.since else None

    try:
        summary = migrate_helicone(
            helicone_api_key=args.helicone_api_key,
            tokensentinel_endpoint=args.tokensentinel_endpoint,
            tokensentinel_api_key=args.tokensentinel_api_key,
            project=args.project,
            since=since,
            dry_run=args.dry_run,
            on_progress=_make_progress(),
        )
    except Exception as exc:
        sys.stderr.write(f"[migrate] error: {exc}\n")
        return 1

    _render_summary(summary, dry_run=args.dry_run)
    return 0


def _run_langfuse(args: argparse.Namespace) -> int:
    rc = _check_cloud_creds(args)
    if rc is not None:
        return rc
    since = _parse_since(args.since) if args.since else None

    try:
        summary = migrate_langfuse(
            public_key=args.langfuse_public_key,
            secret_key=args.langfuse_secret_key,
            tokensentinel_endpoint=args.tokensentinel_endpoint,
            tokensentinel_api_key=args.tokensentinel_api_key,
            project=args.project,
            since=since,
            base_url=args.langfuse_base_url,
            dry_run=args.dry_run,
            on_progress=_make_progress(),
        )
    except Exception as exc:
        sys.stderr.write(f"[migrate] error: {exc}\n")
        return 1

    _render_summary(summary, dry_run=args.dry_run)
    return 0


def _run_langsmith(args: argparse.Namespace) -> int:
    rc = _check_cloud_creds(args)
    if rc is not None:
        return rc
    since = _parse_since(args.since) if args.since else None

    try:
        summary = migrate_langsmith(
            api_key=args.langsmith_api_key,
            tokensentinel_endpoint=args.tokensentinel_endpoint,
            tokensentinel_api_key=args.tokensentinel_api_key,
            project=args.project,
            since=since,
            base_url=args.langsmith_base_url,
            dry_run=args.dry_run,
            on_progress=_make_progress(),
        )
    except Exception as exc:
        sys.stderr.write(f"[migrate] error: {exc}\n")
        return 1

    _render_summary(summary, dry_run=args.dry_run)
    return 0


def _render_summary(summary: Any, *, dry_run: bool) -> None:
    """Print the human-readable summary block.

    Accepts any ``MigrationSummary``-shaped object — all three importers
    expose the same dataclass fields, so the renderer is duck-typed.
    """
    if not summary.events_by_type:
        # All-zero summary — print a single line so the user sees the
        # zero-leak signal explicitly. This happens when the customer's
        # source traffic is genuinely clean (great!) or when --since cut
        # off all data (warn-worthy).
        if summary.fetched_traces == 0:
            sys.stdout.write(
                "[migrate] No traces fetched. Check --since and your provider credentials.\n"
            )
            return
        sys.stdout.write(
            "[migrate] 0 leak events fired. Your traffic was clean over the import window.\n"
        )
        return

    # Per-leak-type firings — left-aligned for readability. Sort by name so
    # the output is deterministic regardless of dict ordering.
    for leak_type in sorted(summary.events_by_type):
        count = summary.events_by_type[leak_type]
        sys.stdout.write(
            f"[migrate]   {leak_type:<20} {count} firing" + ("s" if count != 1 else "") + "\n"
        )
    other_known = {"tool_loop", "retry_storm", "model_misroute", "embedding_waste"}
    others_count = sum(v for k, v in summary.events_by_type.items() if k not in other_known)
    if others_count:
        sys.stdout.write(f"[migrate]   {'(others)':<20} {others_count} firings\n")

    if dry_run:
        sys.stdout.write(
            f"[migrate] {summary.fired_events} leak events would be backfilled "
            "(dry-run, not posted)\n"
        )
    else:
        sys.stdout.write(
            f"[migrate] {summary.backfill_accepted} events backfilled to "
            f"TokenSentinel cloud "
            f"({summary.backfill_rejected} rejected).\n"
        )

    # Cost-saved estimate. Per the founder spec: estimated_burn per event,
    # summed per leak type. Sorted by descending burn so the highest-impact
    # leak class lands at the top.
    sys.stdout.write("[migrate]\n")
    sys.stdout.write("[migrate] Estimated cost saved if these had been intervened:\n")
    sorted_burn = sorted(summary.burn_by_type.items(), key=lambda kv: kv[1], reverse=True)
    for leak_type, burn in sorted_burn:
        if burn <= 0:
            continue
        label = f"{leak_type} savings:"
        sys.stdout.write(f"[migrate]   {label:<25}  ${burn:.2f}\n")
    sys.stdout.write(f"[migrate]   {'total:':<25}  ${summary.estimated_burn_total_usd:.2f}\n")

    if dry_run:
        sys.stdout.write("[migrate]\n")
        sys.stdout.write(
            "[migrate] Re-run without --dry-run to backfill events to TokenSentinel cloud.\n"
        )


def _parse_since(raw: str) -> Any:
    """Parse the ``--since`` CLI argument as a UTC datetime."""
    s = raw.strip()
    # Accept bare-date form (YYYY-MM-DD) by appending T00:00:00Z. Anything
    # else falls through to fromisoformat, which handles the various
    # canonical ISO-8601 shapes.
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        s = s + "T00:00:00+00:00"
    elif s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError as exc:
        raise SystemExit(
            f"[migrate] error: --since must be an ISO-8601 date or datetime (got {raw!r}): {exc}"
        ) from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


if __name__ == "__main__":
    raise SystemExit(main())
