"""Module entry point for ``python -m tokensentinel_migrate``."""

from __future__ import annotations

from tokensentinel_migrate.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
