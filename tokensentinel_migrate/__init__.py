"""tokensentinel-migrate — importers from LLM observability tools to TokenSentinel.

Public surface:

- :func:`migrate_helicone` — fetch traces from Helicone, replay them through
  TokenSentinel rules, and (optionally) backfill the resulting leak events to
  a TokenSentinel cloud project.
- :class:`HeliconeImporter` — the importer object underneath, exposed for
  programmatic / scripted use.
- :class:`MigrationSummary` — the structured result returned by every
  importer (event counts, dollar savings, error tallies).

The CLI entry point is :func:`tokensentinel_migrate.cli.main` (registered as
``tokensentinel-migrate`` in ``pyproject.toml``).
"""

from __future__ import annotations

from tokensentinel_migrate.helicone import (
    HeliconeImporter,
    MigrationSummary,
    migrate_helicone,
)

__version__ = "0.1.0"
__all__ = [
    "HeliconeImporter",
    "MigrationSummary",
    "migrate_helicone",
]
