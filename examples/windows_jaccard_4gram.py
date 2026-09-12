"""Run the simple sphncs baseline on the first Windows LogHub records.

The Zenodo tarball is streamed, so this script stops reading as soon as it has
the requested number of newline-delimited log records instead of unpacking the
full archive.
"""

from __future__ import annotations

import argparse
import json
import tarfile
from functools import partial
from pathlib import Path
from urllib.request import urlopen

from sphncs import SphncsClusterer
from sphncs.distances import char_ngram_jaccard


DEFAULT_URL = "https://zenodo.org/record/3227177/files/Windows.tar.gz"


def read_first_log_records(url: str, limit: int) -> tuple[list[str], str]:
    """Read the first ``limit`` nonempty records from the first log file."""
    with urlopen(url) as response, tarfile.open(fileobj=response, mode="r|gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.lower().endswith(".log"):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            records: list[str] = []
            for line in extracted:
                value = line.decode("utf-8", errors="replace").rstrip("\r\n")
                if value:
                    records.append(value)
                if len(records) == limit:
                    return records, member.name
    raise RuntimeError(f"No .log member with {limit} nonempty records was found in {url}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument(
        "--log-filter",
        dest="log_filters",
        action="append",
        default=[],
        help="Preprocessing filter to apply; repeat for each selection (for example: --log-filter timestamp).",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/windows-jaccard-4gram-first1000.json")
    )
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")

    records, member_name = read_first_log_records(args.url, args.limit)
    metric = partial(char_ngram_jaccard, ngram_size=4)
    model = SphncsClusterer(
        metric=metric,
        length_partitioning=False,
        clustering_mode="single",
        n_embeddings=1,
        random_state=7,
        log_filters=args.log_filters,
    ).fit(records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "source_url": args.url,
        "archive_member": member_name,
        "records": len(records),
        "metric": "char_ngram_jaccard",
        "ngram_size": 4,
        "length_partitioning": False,
        "clustering_mode": "single",
        "n_embeddings": 1,
        "random_state": 7,
        "log_filters": model.preprocessor_.filters,
        "n_clusters": int(model.n_clusters_),
        "representative_indices": model.representative_indices_.tolist(),
        "representatives": model.representatives_,
        "raw_representatives": model.raw_representatives_,
        "cluster_sizes": [int((model.labels_ == label).sum()) for label in range(model.n_clusters_)],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
