# Contributing to TokenSentinel Migrate

Thank you for your interest in contributing to TokenSentinel Migrate! This tool helps teams migrate their historical trace data from LLM observability platforms like Helicone, Langfuse, and LangSmith into TokenSentinel.

This document outlines guidelines for setting up a development environment, running tests, adding new importers, and submitting contributions.

## Code of Conduct

We expect all contributors to maintain a welcoming, respectful, and friendly environment focused on collaborative progress and high-quality software.

## Getting Started

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- Support for Linux, macOS, or Windows (via WSL)

### 2. Setup Development Environment
Clone the repository and install the package in editable mode with development dependencies:

```bash
git clone https://github.com/tokensentinel/tokensentinel-migrate-python.git
cd tokensentinel-migrate-python

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install package and dev requirements
pip install -e ".[dev]"
```

## Development Workflow

### 1. Code Style and Formatting
We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting. Ensure your changes pass style checks before submitting:

```bash
# Check code style and run lints
python3 -m ruff check tokensentinel_migrate tests

# Format files
python3 -m ruff format tokensentinel_migrate tests
```

### 2. Running Tests
Tests are implemented using standard `pytest`. The test suite uses unittest mocks to simulate provider API responses so that no real network calls or API keys are required during testing:

```bash
# Run all tests
python3 -m pytest
```

You can run individual test files:
```bash
# Example: Run Helicone importer tests
python3 -m pytest tests/test_helicone.py
```

## How to Add a New Importer

Importers are modular and structured as separate commands/files:

1. **Add Importer Module**: Create a new file in `tokensentinel_migrate/<provider>.py`.
2. **Implement Fetch and Normalize logic**:
   - Fetch traces/requests from the source platform's API using pagination (e.g. cursors or page offsets). Use standard libraries (`urllib.request`) to keep runtime dependencies to a minimum.
   - Normalize the fetched records into TokenSentinel's standard `CallRecord` schema.
3. **Replay Rules**: Map normalized `CallRecord`s through `_retroactive.py` to evaluate the 8 core waste rules.
4. **Register CLI Command**: Update `tokensentinel_migrate/cli.py` to register the new subcommand and arguments.
5. **Add Tests**: Write mocked unit tests in `tests/test_<provider>.py` to verify pagination, field mapping, rule firing, and backfill POST calls.

## Security & Privacy Guidelines

- **API Keys**: Importers must never hardcode API keys, tokens, or endpoints. Accept credentials strictly via command-line arguments and environment variables.
- **Redaction**: Normalized trace payloads must handle sensitive user content carefully.
- **Mocking**: Ensure all new tests use mocks to intercept network requests (`urllib.request.urlopen`). Do not make live API queries inside the test suite.

## Submitting a Pull Request

1. Fork the repository and branch from `main`.
2. Make your changes and write unit tests for any new behavior or bug fixes.
3. Verify that all formatting, linting, and tests pass:
   ```bash
   python3 -m ruff check tokensentinel_migrate tests
   python3 -m ruff format tokensentinel_migrate tests
   python3 -m pytest
   ```
4. Push your branch to your fork and submit a Pull Request. Provide a concise summary of the additions and why they are necessary.

## Support

If you have questions, face issues with a migration, or want to suggest new importers, please open an issue in this repository or contact us at [shakyasmreta@gmail.com](mailto:shakyasmreta@gmail.com).
