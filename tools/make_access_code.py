#!/usr/bin/env python3
"""Issue a personal access code for one advisor (MCP_AUTH=1 OAuth login).

Usage:
    uv run python tools/make_access_code.py fargona_sh

Prints the code ONCE (hand it to the advisor over a private channel) and the
``name:sha256hex`` entry to append to MCP_ACCESS_CODES. Nothing is written to
disk; only the hash goes into the server's configuration.

Revoke an advisor by deleting their entry from MCP_ACCESS_CODES and
redeploying. Running this again for the same name issues a replacement code:
swap the entry, and the old code and all its tokens stop working.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from telegram_mcp.auth import (  # noqa: E402
    AuthConfigError,
    generate_access_code,
    hash_access_code,
    parse_access_codes,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("name", help="advisor label, e.g. fargona_sh")
    args = parser.parse_args(argv)

    code = generate_access_code()
    entry = f"{args.name}:{hash_access_code(code)}"
    try:
        parse_access_codes(entry)  # same validation the server applies
    except AuthConfigError as exc:
        parser.error(str(exc))

    print(f"Access code for {args.name} (shown once, give it to the advisor):")
    print(f"    {code}")
    print()
    print("Append this entry to MCP_ACCESS_CODES (comma-separated), then redeploy:")
    print(f"    {entry}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
