from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .compressor import TokenCompressor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compress Chinese sentences into shorter equivalent text.")
    parser.add_argument("text", nargs="?", help="sentence to compress")
    parser.add_argument("-r", "--ratio", type=float, default=0.65, help="target compression ratio, default: 0.65")
    parser.add_argument("-p", "--profile", help="JSON rule profile")
    parser.add_argument("--learn", help="CSV file with columns: original, compressed")
    parser.add_argument("--save-profile", help="path to save learned profile")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    compressor = TokenCompressor.from_profile(args.profile) if args.profile else TokenCompressor()

    if args.learn:
        if not args.save_profile:
            raise SystemExit("--learn requires --save-profile")
        pairs = _read_pairs(args.learn)
        profile = compressor.learn_profile(pairs)
        compressor.save_profile(args.save_profile, profile)
        print(f"saved profile: {args.save_profile}")
        return

    if not args.text:
        raise SystemExit("missing text")

    result = compressor.compress(args.text, target_ratio=args.ratio)
    print(result.compressed)
    print(f"tokens: {result.original_tokens} -> {result.compressed_tokens}, ratio={result.compression_ratio}")
    if result.removed:
        print("removed:", " / ".join(result.removed))


def _read_pairs(path: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if "original" not in reader.fieldnames or "compressed" not in reader.fieldnames:
            raise SystemExit("CSV must contain original and compressed columns")
        for row in reader:
            rows.append((row["original"], row["compressed"]))
    return rows


if __name__ == "__main__":
    main()
