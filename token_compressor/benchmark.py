from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .compressor import CompressionResult, TokenCompressor


@dataclass(frozen=True)
class BenchmarkCase:
    text: str
    mode: str = "balanced"
    keywords: tuple[str, ...] = ()
    max_ratio: float = 0.75
    min_anchor_recall: float = 0.85


@dataclass(frozen=True)
class BenchmarkFailure:
    case: BenchmarkCase
    result: CompressionResult
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class BenchmarkReport:
    count: int
    passed: int
    failed: int
    avg_ratio: float
    avg_anchor_recall: float
    warning_rate: float
    token_counter: str
    failures: tuple[BenchmarkFailure, ...]

    def to_dict(self, include_failures: bool = True) -> dict[str, object]:
        data: dict[str, object] = {
            "count": self.count,
            "passed": self.passed,
            "failed": self.failed,
            "avg_ratio": self.avg_ratio,
            "avg_anchor_recall": self.avg_anchor_recall,
            "warning_rate": self.warning_rate,
            "token_counter": self.token_counter,
        }
        if include_failures:
            data["failures"] = [
                {
                    "text": failure.case.text,
                    "compressed": failure.result.compressed,
                    "mode": failure.case.mode,
                    "ratio": failure.result.compression_ratio,
                    "anchor_recall": failure.result.anchor_recall,
                    "warnings": failure.result.warnings,
                    "reasons": failure.reasons,
                }
                for failure in self.failures
            ]
        return data


def load_benchmark(path: str | Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or "text" not in reader.fieldnames:
            raise ValueError("benchmark CSV must contain a text column")

        for row in reader:
            text = (row.get("text") or "").strip()
            if not text:
                continue
            cases.append(
                BenchmarkCase(
                    text=text,
                    mode=(row.get("mode") or "balanced").strip() or "balanced",
                    keywords=_split_keywords(row.get("keywords") or ""),
                    max_ratio=float(row.get("max_ratio") or 0.75),
                    min_anchor_recall=float(row.get("min_anchor_recall") or 0.85),
                )
            )
    return cases


def run_benchmark(compressor: TokenCompressor, cases: list[BenchmarkCase]) -> BenchmarkReport:
    failures: list[BenchmarkFailure] = []
    results: list[CompressionResult] = []

    for case in cases:
        result = compressor.compress(case.text, mode=case.mode, keywords=case.keywords)
        results.append(result)

        reasons: list[str] = []
        if result.compression_ratio > case.max_ratio:
            reasons.append(f"ratio>{case.max_ratio}")
        if result.anchor_recall < case.min_anchor_recall:
            reasons.append(f"anchor_recall<{case.min_anchor_recall}")
        if result.warnings:
            reasons.append("warnings")
        if reasons:
            failures.append(BenchmarkFailure(case, result, tuple(reasons)))

    if not results:
        return BenchmarkReport(0, 0, 0, 0.0, 0.0, 0.0, compressor.token_counter.name, ())

    return BenchmarkReport(
        count=len(results),
        passed=len(results) - len(failures),
        failed=len(failures),
        avg_ratio=round(sum(item.compression_ratio for item in results) / len(results), 3),
        avg_anchor_recall=round(sum(item.anchor_recall for item in results) / len(results), 3),
        warning_rate=round(sum(1 for item in results if item.warnings) / len(results), 3),
        token_counter=compressor.token_counter.name,
        failures=tuple(failures),
    )


def _split_keywords(value: str) -> tuple[str, ...]:
    if not value:
        return ()
    normalized = value.replace("，", ",").replace("|", ",")
    return tuple(item.strip() for item in normalized.split(",") if item.strip())
