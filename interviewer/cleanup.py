"""Delete interview data older than the retention period.

    python -m interviewer.cleanup --days 30 --data-dir /data
"""

from __future__ import annotations

import argparse
from pathlib import Path

from interviewer.session.store import SessionStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Delete interview sessions older than N days.")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--data-dir", type=Path, default=Path("/data"))
    args = parser.parse_args(argv)
    deleted = SessionStore(args.data_dir).cleanup(args.days)
    for session_id in deleted:
        print(f"deleted {session_id}")
    print(f"{len(deleted)} session(s) older than {args.days} days deleted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
