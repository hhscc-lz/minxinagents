"""Web-friendly entrypoint for serving the Textual CLI."""

from __future__ import annotations

import asyncio

from deepagents_cli.main import run_textual_cli_async


def main() -> int:
    """Run the Textual CLI through the standard interactive launcher."""
    result = asyncio.run(run_textual_cli_async("agent"))
    return result.return_code


if __name__ == "__main__":
    raise SystemExit(main())
