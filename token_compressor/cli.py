from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .benchmark import load_benchmark, run_benchmark
from .compressor import TokenCompressor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compress Chinese sentences into shorter equivalent text.")
    parser.add_argument("text", nargs="?", help="sentence to compress")
    parser.add_argument(
        "-m",
        "--mode",
        choices=("safe", "balanced", "aggressive"),
        default="balanced",
        help="compression mode, default: balanced",
    )
    parser.add_argument("-r", "--ratio", type=float, help="override target compression ratio")
    parser.add_argument("-p", "--profile", help="JSON rule profile")
    parser.add_argument("-c", "--config", help="JSON config with custom drop phrases, replacements, and templates")
    parser.add_argument("-k", "--keyword", action="append", default=[], help="keyword that must be preserved")
    parser.add_argument(
        "--token-counter",
        choices=("auto", "coarse", "tiktoken"),
        default="auto",
        help="token counter backend, default: auto",
    )
    parser.add_argument("--tiktoken-encoding", default="cl100k_base", help="tiktoken encoding name")
    parser.add_argument("--input-file", help="UTF-8 text file, one sentence per line")
    parser.add_argument("--paragraph", action="store_true", help="treat input text or input file as one paragraph")
    parser.add_argument("--json", action="store_true", help="print JSON lines")
    parser.add_argument("--details", action="store_true", help="print compression diagnostics")
    parser.add_argument("--diff", action="store_true", help="print token-level delete/replace operations")
    parser.add_argument("--learn", help="CSV file with columns: original, compressed")
    parser.add_argument("--save-profile", help="path to save learned profile")
    parser.add_argument("--evaluate", help="UTF-8 text file used to print average compression metrics")
    parser.add_argument("--benchmark", help="CSV benchmark file with text, mode, keywords, max_ratio columns")
    parser.add_argument("--fail-on-benchmark", action="store_true", help="exit with code 1 when benchmark fails")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.profile and args.config:
        raise SystemExit("--profile and --config cannot be used together")

    if args.config:
        compressor = TokenCompressor.from_config(
            args.config,
            token_counter=args.token_counter,
            tiktoken_encoding=args.tiktoken_encoding,
        )
    elif args.profile:
        compressor = TokenCompressor.from_profile(
            args.profile,
            token_counter=args.token_counter,
            tiktoken_encoding=args.tiktoken_encoding,
        )
    else:
        compressor = TokenCompressor(token_counter=args.token_counter, tiktoken_encoding=args.tiktoken_encoding)

    if args.learn:
        if not args.save_profile:
            raise SystemExit("--learn requires --save-profile")
        pairs = _read_pairs(args.learn)
        profile = compressor.learn_profile(pairs)
        compressor.save_profile(args.save_profile, profile)
        print(f"saved profile: {args.save_profile}")
        return

    if args.evaluate:
        texts = _read_lines(args.evaluate)
        print(json.dumps(compressor.evaluate(texts, target_ratio=args.ratio, mode=args.mode), ensure_ascii=False))
        return

    if args.benchmark:
        report = run_benchmark(compressor, load_benchmark(args.benchmark))
        print(json.dumps(report.to_dict(include_failures=True), ensure_ascii=False, indent=2))
        if args.fail_on_benchmark and report.failed:
            raise SystemExit(1)
        return

    if args.input_file:
        if args.paragraph:
            text = Path(args.input_file).read_text(encoding="utf-8-sig")
            result = compressor.compress_paragraph(text, target_ratio=args.ratio, mode=args.mode, keywords=args.keyword)
            _print_paragraph_result(result, as_json=args.json, details=args.details, show_diff=args.diff)
            return
        for text in _read_lines(args.input_file):
            result = compressor.compress(text, target_ratio=args.ratio, mode=args.mode, keywords=args.keyword)
            _print_result(result, as_json=args.json, details=args.details, show_diff=args.diff)
        return

    if not args.text:
        raise SystemExit("missing text")

    if args.paragraph:
        result = compressor.compress_paragraph(args.text, target_ratio=args.ratio, mode=args.mode, keywords=args.keyword)
        _print_paragraph_result(result, as_json=args.json, details=args.details, show_diff=args.diff)
    else:
        result = compressor.compress(args.text, target_ratio=args.ratio, mode=args.mode, keywords=args.keyword)
        _print_result(result, as_json=args.json, details=True if args.details else False, show_diff=args.diff)


def _read_pairs(path: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if "original" not in reader.fieldnames or "compressed" not in reader.fieldnames:
            raise SystemExit("CSV must contain original and compressed columns")
        for row in reader:
            rows.append((row["original"], row["compressed"]))
    return rows


def _read_lines(path: str) -> list[str]:
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _print_result(result, as_json: bool, details: bool, show_diff: bool) -> None:
    if as_json:
        print(
            json.dumps(
                {
                    "original": result.original,
                    "compressed": result.compressed,
                    "mode": result.mode,
                    "original_tokens": result.original_tokens,
                    "compressed_tokens": result.compressed_tokens,
                    "compression_ratio": result.compression_ratio,
                    "anchor_recall": result.anchor_recall,
                    "preserved_terms": result.preserved_terms,
                    "warnings": result.warnings,
                    "removed": result.removed,
                    "candidates_considered": result.candidates_considered,
                    "token_counter": result.token_counter,
                    "diff": _diff_to_dict(result.diff),
                },
                ensure_ascii=False,
            )
        )
        return

    print(result.compressed)
    if details:
        print(
            "tokens: "
            f"{result.original_tokens} -> {result.compressed_tokens}, "
            f"ratio={result.compression_ratio}, "
            f"mode={result.mode}, "
            f"anchor_recall={result.anchor_recall}, "
            f"counter={result.token_counter}"
        )
        if result.removed:
            print("removed:", " / ".join(result.removed))
        if result.warnings:
            print("warnings:", " / ".join(result.warnings))
    if show_diff:
        _print_diff(result.diff)


def _print_paragraph_result(result, as_json: bool, details: bool, show_diff: bool) -> None:
    if as_json:
        print(
            json.dumps(
                {
                    "original": result.original,
                    "compressed": result.compressed,
                    "original_tokens": result.original_tokens,
                    "compressed_tokens": result.compressed_tokens,
                    "compression_ratio": result.compression_ratio,
                    "removed_sentences": result.removed_sentences,
                    "warnings": result.warnings,
                    "token_counter": result.token_counter,
                    "diff": _diff_to_dict(result.diff),
                    "sentences": [
                        {
                            "original": item.original,
                            "compressed": item.compressed,
                            "ratio": item.compression_ratio,
                            "anchor_recall": item.anchor_recall,
                            "warnings": item.warnings,
                            "diff": _diff_to_dict(item.diff),
                        }
                        for item in result.sentence_results
                    ],
                },
                ensure_ascii=False,
            )
        )
        return

    print(result.compressed)
    if details:
        print(
            "tokens: "
            f"{result.original_tokens} -> {result.compressed_tokens}, "
            f"ratio={result.compression_ratio}, "
            f"sentences={len(result.sentence_results)}, "
            f"removed_sentences={len(result.removed_sentences)}, "
            f"counter={result.token_counter}"
        )
        if result.removed_sentences:
            print("removed sentences:", " / ".join(result.removed_sentences))
        if result.warnings:
            print("warnings:", " / ".join(result.warnings))
    if show_diff:
        _print_diff(result.diff)


def _diff_to_dict(diff) -> list[dict[str, str]]:
    return [{"op": item.op, "source": item.source, "target": item.target} for item in diff]


def _print_diff(diff) -> None:
    if not diff:
        print("diff: no changes")
        return
    print("diff:")
    for item in diff:
        if item.op in ("delete", "delete_sentence"):
            print(f"  - {item.source}")
        elif item.op == "insert":
            print(f"  + {item.target}")
        else:
            print(f"  ~ {item.source} -> {item.target}")


if __name__ == "__main__":
    main()
