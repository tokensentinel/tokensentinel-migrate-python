# TokenSentinel Migrate (tokensentinel-migrate)

Migrate from LLM observability tools — Helicone, Langfuse, and LangSmith — into [TokenSentinel](https://github.com/tokensentinel/tokensentinel-sdk-python) without losing the months of trace data you've already accumulated.

## Why this exists

In March 2026, [Mintlify acquired Helicone](https://www.mintlify.com/blog/mintlify-acquires-helicone). Mintlify's interest is the AI documentation play; Helicone is now in maintenance mode — security patches and bug fixes only, [no new integrations, no new analytics, no roadmap](https://dev.to/torrixai/helicone-is-now-in-maintenance-mode-here-is-how-to-switch-to-a-self-hosted-alternative-in-5-4li0). The 16,000 organisations that built on Helicone are all looking for somewhere else to go.

If you were one of them, this tool is your bridge. In one command it pulls your Helicone trace history, replays it through TokenSentinel's eight waste detection rules, and shows you the dollars you would have saved if you'd had intervention turned on. Then, if you want, it backfills the resulting events to your TokenSentinel cloud project so the dashboard reflects history-plus-now from day one.

It is **MIT-licensed**, **stdlib-only** apart from the TokenSentinel SDK itself, and **runs entirely on your machine** — your Helicone API key never leaves your laptop.

## 5-minute migration

```bash
pip install tokensentinel-migrate

python -m tokensentinel_migrate helicone \
    --helicone-api-key sk-helicone-... \
    --tokensentinel-endpoint https://api.tokensentinel.dev \
    --tokensentinel-api-key tsk_... \
    --project my-agent \
    --since 2026-04-09 \
    --dry-run
```

Sample output:

```
[migrate] Fetching Helicone traces since 2026-04-09...
[migrate]   page 1: 100 requests
[migrate]   page 2: 100 requests
[migrate]   page 3: 47 requests
[migrate] Fetched 247 traces (12 sessions inferred from heliconeproperty Helicone-Session-Id)
[migrate] Running TokenSentinel rules retroactively...
[migrate]   tool_loop:           3 firings
[migrate]   retry_storm:         1 firing
[migrate]   model_misroute:      8 firings
[migrate]   embedding_waste:     0 firings
[migrate]   (others):            0 firings
[migrate] 12 leak events would be backfilled (dry-run, not posted)
[migrate]
[migrate] Estimated cost saved if these had been intervened:
[migrate]   tool_loop savings:        $0.83
[migrate]   retry_storm savings:      $0.21
[migrate]   model_misroute savings:   $4.42
[migrate]   total:                    $5.46
[migrate]
[migrate] Re-run without --dry-run to backfill events to TokenSentinel cloud.
```

Re-running without `--dry-run` POSTs each event to the cloud's backfill endpoint so the dashboard's "tokens saved this week" counter reflects what TokenSentinel would have caught had it been wired in across the import window.

## What gets migrated

For each Helicone request the importer pulls:

| Helicone field | TokenSentinel `CallRecord` field |
| --- | --- |
| `provider`, `model` | `provider`, `model` |
| `prompt_tokens`, `completion_tokens` (or nested `usage.*_tokens`) | matching fields |
| `latency_ms` | `latency_ms` |
| `created_at` | `timestamp` (UTC-normalised) |
| `request_id` | `request_hash` (used by `retry_storm` for dedup) |
| `properties["Helicone-Session-Id"]` &rarr; `properties["session_id"]` &rarr; `request_id` | `session_id` |
| `body.messages` or `prompt` | `raw_request.messages` |
| `body.input` | `raw_request.input` (drives `embedding_waste`) |

Embedding-shaped models (anything with `embedding` in the name) are routed to `embeddings.create` so the `embedding_waste` rule fires correctly.

## What you get back

Each Helicone request that triggered a TokenSentinel rule is converted into a `LeakEvent` and POSTed to your cloud project at `<endpoint>/v1/events:backfill?project=<project>` with the **original timestamp preserved**. That last detail matters: without it, the dashboard would attribute every backfilled event to "today" and the savings counter would double-count migrated history as live activity. With it, the dashboard timeline reflects the truth — these leaks happened on the days Helicone says they happened.

The CLI also surfaces a per-leak-type **dollar savings estimate** for the import window, summed from each event's `estimated_burn` field. That's the number to forward to your CFO.

## Helicone API quirks worth knowing

A few footnotes from the Helicone integration:

- **`POST /v1/request/query`, not GET.** Pagination + filter both ride in the JSON body. `offset` and `limit` are top-level keys; the SDK uses `limit=100` per page (the maximum at the time of the Mintlify acquisition).
- **Timestamps come as `Z`-suffixed ISO-8601.** `datetime.fromisoformat` on Python 3.10 needs the `Z` swapped for `+00:00`; we do that.
- **`properties` casing.** Helicone's official SDK ships `Helicone-Session-Id` (mixed case); some community SDKs ship `session_id`. We check both, in that order, then fall back to `request_id` for one-call sessions.
- **`Retry-After` header.** Sometimes seconds-as-integer, sometimes HTTP-date. We honour the integer form; HTTP-date callers get a 5-second default backoff. Both forms cap out at 60 seconds so a misbehaving deploy can't strand the CLI.
- **Non-2xx behaviour.** 401/403 abort immediately (check your key); 429 sleeps and retries up to six times in a row before giving up; everything else is non-retryable and surfaces in stderr.

## Langfuse

Langfuse is the largest OSS LLM observability project and the second migration target after Helicone. The Langfuse importer pulls every `GENERATION` observation from your traces and replays them through the same eight waste rules.

```bash
python -m tokensentinel_migrate langfuse \
    --langfuse-public-key pk-lf-... \
    --langfuse-secret-key sk-lf-... \
    --langfuse-base-url https://cloud.langfuse.com \
    --tokensentinel-endpoint https://api.tokensentinel.dev \
    --tokensentinel-api-key tsk_... \
    --project my-agent \
    --since 2026-04-09 \
    --dry-run
```

Self-hosted Langfuse users point `--langfuse-base-url` at their own deployment — the default is `https://cloud.langfuse.com`.

Langfuse gotchas:

- **Two-key auth.** Langfuse uses HTTP Basic with the public key as the username and the secret key as the password. Both are required; the importer aborts with a clean error if either is missing.
- **Only `type=="GENERATION"` observations are imported.** `SPAN` / `EVENT` rows don't represent real LLM calls and the rule engine has no meaningful interpretation for them — they're dropped during normalisation.
- **`usage.unit == "CHARACTERS"` zeros the token count.** Langfuse customers who never wired token counting see a degraded cost estimate (the CallRecord still propagates so non-token rules like `tool_loop` and `retry_storm` fire correctly).
- **Embedding detection is lossy.** Langfuse doesn't preserve the SDK method, so every CallRecord lands as `messages.create`. The `embedding_waste` rule under-fires on Langfuse imports relative to Helicone — a known tradeoff documented in the founder spec.

## LangSmith

LangSmith is LangChain's hosted observability product and the default trace destination for any LangChain / LangGraph agent. The importer queries the `/runs/query` cursor-paginated endpoint.

```bash
python -m tokensentinel_migrate langsmith \
    --langsmith-api-key ls__... \
    --langsmith-base-url https://api.smith.langchain.com \
    --tokensentinel-endpoint https://api.tokensentinel.dev \
    --tokensentinel-api-key tsk_... \
    --project my-agent \
    --since 2026-04-09 \
    --dry-run
```

Enterprise LangSmith tenants point `--langsmith-base-url` at their per-tenant URL; the default is `https://api.smith.langchain.com`.

LangSmith gotchas:

- **Only `run_type=="llm"` runs are imported.** `chain` / `tool` / `retriever` runs are dropped — the rule engine reads `CallRecord.tool_calls` from the LLM run's structured output instead, which captures the same signal more reliably.
- **Token counts live in two places.** Newer LangSmith ships `prompt_tokens` / `completion_tokens` at the top level; older versions stash them under `extra.invocation_params.usage`. The importer checks both, in that order, before falling back to (0, 0).
- **Cursor pagination, not page numbers.** The importer keeps re-POSTing the `cursor` from the previous response until the server hands back `cursors.next == null`. There's no way to know up front how many pages a date range will produce.
- **Provider inference is heuristic.** LangSmith's `_type` field (`anthropic-chat`, `chat-openai`, etc.) drives a regex match; the model-name fallback (`claude` → anthropic, `gpt` → openai, …) catches non-standard `_type` values.

## Roadmap

| Importer | Status | When |
| --- | --- | --- |
| Helicone | shipping in v0.1.0 | now |
| Langfuse | shipping in v0.2.0 | now |
| LangSmith | shipping in v0.2.0 | now |

Each importer is a separate subcommand under `python -m tokensentinel_migrate` and a separate module under `tokensentinel_migrate/`. The shared infrastructure (`_backfill.py` and `_retroactive.py`) is provider-agnostic; adding a new importer is a couple-of-hundred lines of `fetch + normalise + pagination` glue.

## Development

```bash
git clone https://github.com/tokensentinel/tokensentinel-migrate-python
cd tokensentinel-migrate-python
pip install -e ".[dev]"
python -m pytest
python -m ruff check tokensentinel_migrate tests
```

The test suite uses `unittest.mock.patch('urllib.request.urlopen', ...)` to inject canned Helicone responses and to verify the cloud-side backfill payload — no live network calls in CI.

## License

MIT. See [LICENSE](./LICENSE).

## Contact & Support

- The TokenSentinel SDK lives at [github.com/tokensentinel/tokensentinel-sdk-python](https://github.com/tokensentinel/tokensentinel-sdk-python).
- The official website is [tokensentinel.dev](https://tokensentinel.dev).
- For questions, bug reports, or support with a stuck migration, please file an issue in this repository or contact us at [shakyasmreta@gmail.com](mailto:shakyasmreta@gmail.com).
