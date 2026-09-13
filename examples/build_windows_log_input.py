"""Download a prefix of the Windows corpus and save raw and filtered logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sphncs.preprocessing import LogPreprocessor
from windows_jaccard_4gram import DEFAULT_URL, read_first_log_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--limit", type=int, default=1_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")

    raw, archive_member = read_first_log_records(args.url, args.limit)
    processed = LogPreprocessor("all").transform_many(raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "source_url": args.url,
                "archive_member": archive_member,
                "raw": raw,
                "processed": processed,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output} ({len(raw)} records; {len(set(processed))} unique processed templates)")


if __name__ == "__main__":
    main()
