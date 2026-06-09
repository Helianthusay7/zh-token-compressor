from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class CompressionResult:
    original: str
    compressed: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    removed: tuple[str, ...]


class TokenCompressor:
    """A lightweight Chinese sentence compressor.

    It is intentionally small and deterministic: useful for prototypes,
    prompt shortening, and generating training data before replacing it with
    a neural seq2seq model.
    """

    DEFAULT_DROP_PHRASES = (
        "我认为",
        "我觉得",
        "个人感觉",
        "其实",
        "基本上",
        "大概",
        "可能",
        "也许",
        "就是说",
        "也就是说",
        "换句话说",
        "总的来说",
        "从某种程度上来说",
        "在我看来",
        "非常",
        "特别",
        "比较",
        "相当",
        "十分",
        "真的",
        "确实",
        "一定程度上",
        "的话",
        "这个",
        "那个",
        "一些",
        "一下",
    )

    DEFAULT_REPLACEMENTS = {
        "由于": "因",
        "因为": "因",
        "所以": "故",
        "如果": "若",
        "那么": "则",
        "但是": "但",
        "然而": "但",
        "并且": "且",
        "以及": "和",
        "进行": "",
        "需要去": "需",
        "需要": "需",
        "能够": "能",
        "可以": "可",
        "没有办法": "无法",
        "无法进行": "无法",
        "使用": "用",
        "帮助": "助",
        "提高": "提升",
        "降低": "减少",
        "减少": "降",
        "创建": "建",
        "实现": "做",
        "完成": "做完",
    }

    DEFAULT_KEEP_WORDS = (
        "不",
        "没有",
        "无法",
        "必须",
        "应该",
        "需要",
        "禁止",
        "不能",
        "只",
        "才",
        "已",
        "未",
    )

    def __init__(
        self,
        drop_phrases: Iterable[str] | None = None,
        replacements: dict[str, str] | None = None,
        keep_words: Iterable[str] | None = None,
    ) -> None:
        self.drop_phrases = tuple(drop_phrases or self.DEFAULT_DROP_PHRASES)
        self.replacements = dict(self.DEFAULT_REPLACEMENTS)
        if replacements:
            self.replacements.update(replacements)
        self.keep_words = tuple(keep_words or self.DEFAULT_KEEP_WORDS)

    @classmethod
    def from_profile(cls, path: str | Path) -> "TokenCompressor":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            drop_phrases=data.get("drop_phrases"),
            replacements=data.get("replacements"),
            keep_words=data.get("keep_words"),
        )

    def compress(self, text: str, target_ratio: float = 0.65) -> CompressionResult:
        if not 0.2 <= target_ratio <= 1.0:
            raise ValueError("target_ratio must be between 0.2 and 1.0")

        original = self._normalize(text)
        if not original:
            return CompressionResult("", "", 0, 0, 1.0, ())

        candidate, removed = self._drop_phrases(original)
        candidate = self._replace_phrases(candidate)
        candidate = self._dedupe(candidate)
        candidate = self._trim_particles(candidate)
        candidate = self._enforce_target(candidate, target_ratio, removed)
        candidate = self._cleanup(candidate)

        original_tokens = self.count_tokens(original)
        compressed_tokens = self.count_tokens(candidate)
        ratio = compressed_tokens / original_tokens if original_tokens else 1.0

        return CompressionResult(
            original=original,
            compressed=candidate,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=round(ratio, 3),
            removed=tuple(removed),
        )

    def learn_profile(self, pairs: Iterable[tuple[str, str]]) -> dict[str, object]:
        """Infer a small rule profile from original/compressed sentence pairs."""
        dropped: dict[str, int] = {}
        replacements: dict[str, str] = {}

        for original, compressed in pairs:
            original_text = self._normalize(original)
            compressed_text = self._normalize(compressed)

            for source, target in self.DEFAULT_REPLACEMENTS.items():
                if source in original_text and (not target or target in compressed_text):
                    replacements[source] = target

            for phrase in self.DEFAULT_DROP_PHRASES:
                if phrase in original_text and phrase not in compressed_text:
                    dropped[phrase] = dropped.get(phrase, 0) + 1

        learned_drop = [
            word for word, count in sorted(dropped.items(), key=lambda item: (-item[1], -len(item[0])))
            if count >= 1 and word not in self.keep_words
        ]

        return {
            "drop_phrases": learned_drop,
            "replacements": replacements,
            "keep_words": list(self.keep_words),
        }

    def save_profile(self, path: str | Path, profile: dict[str, object]) -> None:
        Path(path).write_text(
            json.dumps(profile, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def count_tokens(self, text: str) -> int:
        return len(self._segment(text))

    def _drop_phrases(self, text: str) -> tuple[str, list[str]]:
        removed: list[str] = []
        candidate = text
        for phrase in sorted(self.drop_phrases, key=len, reverse=True):
            if phrase in candidate and phrase not in self.keep_words:
                candidate = candidate.replace(phrase, "")
                removed.append(phrase)
        return candidate, removed

    def _replace_phrases(self, text: str) -> str:
        candidate = text
        for source, target in sorted(self.replacements.items(), key=lambda item: len(item[0]), reverse=True):
            candidate = candidate.replace(source, target)
        return candidate

    def _dedupe(self, text: str) -> str:
        text = re.sub(r"([，。！？；、,.!?;])\1+", r"\1", text)
        text = re.sub(r"(.{1,4})\1+", r"\1", text)
        return text

    def _trim_particles(self, text: str) -> str:
        text = re.sub(r"[啊呀呢吧嘛啦喔哦]+(?=[，。！？；、,.!?;]|$)", "", text)
        text = re.sub(r"的(?=[，。！？；、,.!?;]|$)", "", text)
        return text

    def _enforce_target(self, text: str, target_ratio: float, removed: list[str]) -> str:
        tokens = self._segment(text)
        target_len = max(1, int(self.count_tokens(text) * target_ratio))
        if len(tokens) <= target_len:
            return text

        weak_tokens = {"的", "地", "得", "了", "着", "过", "就", "都", "还", "也", "又", "再"}
        kept: list[str] = []
        for token in tokens:
            if len(tokens) - len(kept) <= target_len:
                kept.append(token)
                continue
            if token in weak_tokens and token not in self.keep_words:
                removed.append(token)
                continue
            kept.append(token)

        return "".join(kept)

    def _cleanup(self, text: str) -> str:
        text = re.sub(r"\s+", "", text)
        text = re.sub(r"^[，。！？；、,.!?;]+", "", text)
        text = re.sub(r"[，；、,;]+$", "", text)
        text = re.sub(r"([，。！？；、,.!?;])+", lambda match: match.group(0)[0], text)
        return text.strip()

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", "", text.strip())

    def _segment(self, text: str) -> list[str]:
        # A coarse tokenizer: Chinese chars are single tokens; ASCII words/numbers stay grouped.
        return re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]|[^\s]", text)
