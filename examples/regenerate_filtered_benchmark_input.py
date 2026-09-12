"""Regenerate the local processed benchmark view from its preserved raw logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sphncs.preprocessing import LogPreprocessor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON containing the preserved raw logs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    raw = source["raw"]
    output = {"raw": raw, "processed": LogPreprocessor("all").transform_many(raw)}
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({len(raw)} records, {len(set(output['processed']))} unique processed strings)")


if __name__ == "__main__":
    main()
