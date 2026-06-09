from __future__ import annotations

from dataclasses import dataclass
import difflib
import json
import math
import re
from pathlib import Path
from typing import Iterable, Protocol


class TokenCounter(Protocol):
    name: str

    def count(self, text: str) -> int:
        ...


class CoarseTokenCounter:
    name = "coarse"

    def __init__(self, segmenter) -> None:
        self._segmenter = segmenter

    def count(self, text: str) -> int:
        return len(self._segmenter(text))


class TiktokenCounter:
    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        import tiktoken

        self.name = f"tiktoken:{encoding_name}"
        self._encoding = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        return len(self._encoding.encode(text))


@dataclass(frozen=True)
class CompressionMode:
    name: str
    target_ratio: float
    min_anchor_recall: float
    drop_optional_clauses: bool


@dataclass(frozen=True)
class TemplateRule:
    pattern: str
    replacement: str
    label: str


@dataclass(frozen=True)
class DiffOperation:
    op: str
    source: str
    target: str = ""


@dataclass(frozen=True)
class CompressionResult:
    original: str
    compressed: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    removed: tuple[str, ...]
    mode: str = "balanced"
    anchor_recall: float = 1.0
    preserved_terms: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    candidates_considered: int = 0
    token_counter: str = "coarse"
    diff: tuple[DiffOperation, ...] = ()


@dataclass(frozen=True)
class ParagraphCompressionResult:
    original: str
    compressed: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    sentence_results: tuple[CompressionResult, ...]
    removed_sentences: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    token_counter: str = "coarse"
    diff: tuple[DiffOperation, ...] = ()


@dataclass(frozen=True)
class _Candidate:
    text: str
    removed: tuple[str, ...]
    stage: str


class TokenCompressor:
    """A deterministic Chinese sentence compressor.

    The compressor uses a candidate pipeline instead of one hard-coded rewrite:
    it creates several shorter variants, scores them by compression and keyword
    retention, then returns the best candidate that still keeps protected terms.
    """

    MODES = {
        "safe": CompressionMode("safe", target_ratio=0.75, min_anchor_recall=0.92, drop_optional_clauses=False),
        "balanced": CompressionMode("balanced", target_ratio=0.62, min_anchor_recall=0.82, drop_optional_clauses=True),
        "aggressive": CompressionMode("aggressive", target_ratio=0.48, min_anchor_recall=0.70, drop_optional_clauses=True),
    }

    DEFAULT_TEMPLATE_RULES = (
        TemplateRule(r"如果(.{1,40}?)那么(.{1,60})", r"若\1则\2", "如果A那么B->若A则B"),
        TemplateRule(r"由于(.{1,40}?)所以(.{1,60})", r"因\1故\2", "由于A所以B->因A故B"),
        TemplateRule(r"因为(.{1,40}?)所以(.{1,60})", r"因\1故\2", "因为A所以B->因A故B"),
        TemplateRule(
            r"为满足(.{1,40}?)，优化(.{1,40}?)，减少(.{1,40}?)，提升(.{1,50}?)，现启动(.{1,30}?)，依托(.{1,40}?)搭建(.{1,40}?)。",
            r"为满足\1，基于\6开发\7，优化\2，减少\3，提升\4。",
            "业务项目说明模板",
        ),
    )

    DEFAULT_DROP_PHRASES = (
        "我认为",
        "我觉得",
        "个人感觉",
        "在我看来",
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
        "目前来说",
        "简单来说",
    )

    OPTIONAL_CLAUSE_MARKERS = (
        "如果有需要",
        "在不影响使用的情况下",
        "从体验上看",
        "为了更好地",
        "一般来说",
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
        "或者": "或",
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
        "想要": "想",
        "快速地": "快速",
        "更好地": "更好",
        "简单地": "简单",
        "输入内容": "输入",
        "输出结果": "输出",
        "使用量": "用量",
        "表达效率": "表达效率",
        "文本压缩": "压缩文本",
        "日常业务": "业务",
        "现有基础框架": "既有框架",
        "现有流程效率": "流程效率",
        "人工重复操作": "人工重复",
        "减少人工重复": "减少人工重复",
        "数据统一管理": "数据统管",
        "可视化展示能力": "可视化能力",
        "完整可用的业务模块": "业务模块",
        "本功能开发项目": "本项目",
    }

    DEFAULT_KEEP_WORDS = (
        "不",
        "没有",
        "无法",
        "必须",
        "应该",
        "需要",
        "减少",
        "禁止",
        "不能",
        "只",
        "才",
        "已",
        "未",
        "token",
        "Token",
        "tokens",
        "Tokens",
    )

    DEFAULT_DOMAIN_TERMS = (
        "token",
        "Token",
        "tokens",
        "Tokens",
        "提示词",
        "压缩",
        "模型",
        "语义",
        "等效",
        "用户",
        "开发者",
        "功能",
        "工具",
        "输入",
        "输出",
        "文本",
        "句子",
        "数据",
        "训练",
        "学习",
        "规则",
        "接口",
        "配置",
        "效率",
        "用量",
        "保留",
        "关键词",
        "业务",
        "线上化",
        "落地",
        "流程",
        "人工",
        "重复",
        "可视化",
        "框架",
        "模块",
    )

    WEAK_TOKENS = frozenset(("的", "地", "得", "了", "着", "过", "就", "都", "还", "也", "又", "再", "很", "更"))
    PUNCTUATION = "，。！？；、,.!?;"

    def __init__(
        self,
        drop_phrases: Iterable[str] | None = None,
        replacements: dict[str, str] | None = None,
        keep_words: Iterable[str] | None = None,
        domain_terms: Iterable[str] | None = None,
        template_rules: Iterable[TemplateRule | dict[str, str]] | None = None,
        token_counter: str | TokenCounter = "auto",
        tiktoken_encoding: str = "cl100k_base",
    ) -> None:
        self.drop_phrases = tuple(drop_phrases or self.DEFAULT_DROP_PHRASES)
        self.replacements = dict(self.DEFAULT_REPLACEMENTS)
        if replacements:
            self.replacements.update(replacements)
        self.keep_words = tuple(keep_words or self.DEFAULT_KEEP_WORDS)
        self.domain_terms = tuple(domain_terms or self.DEFAULT_DOMAIN_TERMS)
        self.template_rules = self.DEFAULT_TEMPLATE_RULES + tuple(
            self._coerce_template_rule(rule) for rule in (template_rules or ())
        )
        self.lexicon = tuple(sorted(set(self.domain_terms + self.keep_words), key=len, reverse=True))
        self.token_counter = self._build_token_counter(token_counter, tiktoken_encoding)

    @classmethod
    def from_profile(
        cls,
        path: str | Path,
        token_counter: str | TokenCounter = "auto",
        tiktoken_encoding: str = "cl100k_base",
    ) -> "TokenCompressor":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            drop_phrases=data.get("drop_phrases"),
            replacements=data.get("replacements"),
            keep_words=data.get("keep_words"),
            domain_terms=data.get("domain_terms"),
            template_rules=data.get("template_rules"),
            token_counter=token_counter,
            tiktoken_encoding=tiktoken_encoding,
        )

    @classmethod
    def from_config(
        cls,
        path: str | Path,
        token_counter: str | TokenCounter = "auto",
        tiktoken_encoding: str = "cl100k_base",
    ) -> "TokenCompressor":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            drop_phrases=cls.DEFAULT_DROP_PHRASES + tuple(data.get("drop_phrases", ())),
            replacements=data.get("replacements"),
            keep_words=cls.DEFAULT_KEEP_WORDS + tuple(data.get("keep_words", ())),
            domain_terms=cls.DEFAULT_DOMAIN_TERMS + tuple(data.get("domain_terms", ())),
            template_rules=data.get("template_rules"),
            token_counter=token_counter,
            tiktoken_encoding=tiktoken_encoding,
        )

    def compress(
        self,
        text: str,
        target_ratio: float | None = None,
        mode: str = "balanced",
        keywords: Iterable[str] | None = None,
    ) -> CompressionResult:
        config = self._mode_config(mode, target_ratio)
        original = self._normalize(text)
        if not original:
            return CompressionResult("", "", 0, 0, 1.0, (), mode=config.name, token_counter=self.token_counter.name)

        anchors = self._extract_anchors(original, keywords)
        candidates = self._generate_candidates(original, config, anchors)
        best = self._select_candidate(original, candidates, config, anchors)
        compressed = self._cleanup(best.text) or original
        recall = self._anchor_recall(compressed, anchors)
        warnings = self._warnings(original, compressed, anchors, recall, config)

        original_tokens = self.count_tokens(original)
        compressed_tokens = self.count_tokens(compressed)
        ratio = compressed_tokens / original_tokens if original_tokens else 1.0

        return CompressionResult(
            original=original,
            compressed=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=round(ratio, 3),
            removed=best.removed,
            mode=config.name,
            anchor_recall=round(recall, 3),
            preserved_terms=tuple(term for term in anchors if self._anchor_preserved(term, compressed)),
            warnings=warnings,
            candidates_considered=len(candidates),
            token_counter=self.token_counter.name,
            diff=self._build_rule_diff(best.removed) or self._build_diff(original, compressed),
        )

    def batch_compress(
        self,
        texts: Iterable[str],
        target_ratio: float | None = None,
        mode: str = "balanced",
    ) -> list[CompressionResult]:
        return [self.compress(text, target_ratio=target_ratio, mode=mode) for text in texts]

    def compress_paragraph(
        self,
        text: str,
        target_ratio: float | None = None,
        mode: str = "balanced",
        keywords: Iterable[str] | None = None,
    ) -> ParagraphCompressionResult:
        original = text.strip()
        if not original:
            return ParagraphCompressionResult("", "", 0, 0, 1.0, (), token_counter=self.token_counter.name)

        sentence_results: list[CompressionResult] = []
        removed_sentences: list[str] = []
        seen_signatures: set[str] = set()

        for sentence in self._split_sentences(original):
            signature = self._sentence_signature(sentence)
            if signature and signature in seen_signatures:
                removed_sentences.append(sentence)
                continue
            if signature:
                seen_signatures.add(signature)
            sentence_results.append(
                self.compress(sentence, target_ratio=target_ratio, mode=mode, keywords=keywords)
            )

        compressed = self._cleanup_paragraph("".join(result.compressed for result in sentence_results))
        original_tokens = self.count_tokens(original)
        compressed_tokens = self.count_tokens(compressed)
        ratio = compressed_tokens / original_tokens if original_tokens else 1.0
        warnings = tuple(dict.fromkeys(warning for result in sentence_results for warning in result.warnings))
        diff = tuple(DiffOperation("delete_sentence", sentence) for sentence in removed_sentences)
        diff += tuple(operation for result in sentence_results for operation in result.diff)

        return ParagraphCompressionResult(
            original=original,
            compressed=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=round(ratio, 3),
            sentence_results=tuple(sentence_results),
            removed_sentences=tuple(removed_sentences),
            warnings=warnings,
            token_counter=self.token_counter.name,
            diff=diff or self._build_diff(original, compressed),
        )

    def evaluate(
        self,
        texts: Iterable[str],
        target_ratio: float | None = None,
        mode: str = "balanced",
    ) -> dict[str, float]:
        results = self.batch_compress(texts, target_ratio=target_ratio, mode=mode)
        if not results:
            return {"count": 0, "avg_ratio": 0.0, "avg_anchor_recall": 0.0, "warning_rate": 0.0}

        return {
            "count": float(len(results)),
            "avg_ratio": round(sum(item.compression_ratio for item in results) / len(results), 3),
            "avg_anchor_recall": round(sum(item.anchor_recall for item in results) / len(results), 3),
            "warning_rate": round(sum(1 for item in results if item.warnings) / len(results), 3),
        }

    def learn_profile(self, pairs: Iterable[tuple[str, str]]) -> dict[str, object]:
        """Infer a small rule profile from original/compressed sentence pairs."""
        dropped: dict[str, int] = {}
        replacements: dict[str, str] = {}
        domain_terms: set[str] = set(self.domain_terms)

        for original, compressed in pairs:
            original_text = self._normalize(original)
            compressed_text = self._normalize(compressed)

            for source, target in self.DEFAULT_REPLACEMENTS.items():
                if source in original_text and (not target or target in compressed_text):
                    replacements[source] = target

            for phrase in self.DEFAULT_DROP_PHRASES:
                if phrase in original_text and phrase not in compressed_text:
                    dropped[phrase] = dropped.get(phrase, 0) + 1

            for term in self._extract_anchors(compressed_text, keywords=()):
                if len(term) >= 2:
                    domain_terms.add(term)

        learned_drop = [
            word for word, count in sorted(dropped.items(), key=lambda item: (-item[1], -len(item[0])))
            if count >= 1 and word not in self.keep_words
        ]

        return {
            "drop_phrases": learned_drop,
            "replacements": replacements,
            "keep_words": list(self.keep_words),
            "domain_terms": sorted(domain_terms, key=lambda item: (-len(item), item)),
        }

    def save_profile(self, path: str | Path, profile: dict[str, object]) -> None:
        Path(path).write_text(
            json.dumps(profile, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def count_tokens(self, text: str) -> int:
        return self.token_counter.count(text)

    def _build_token_counter(
        self,
        token_counter: str | TokenCounter,
        tiktoken_encoding: str,
    ) -> TokenCounter:
        if not isinstance(token_counter, str):
            return token_counter
        if token_counter == "coarse":
            return CoarseTokenCounter(self._segment)
        if token_counter in ("auto", "tiktoken"):
            try:
                return TiktokenCounter(tiktoken_encoding)
            except ImportError:
                if token_counter == "tiktoken":
                    raise
                return CoarseTokenCounter(self._segment)
        raise ValueError("token_counter must be auto, coarse, tiktoken, or a TokenCounter object")

    def _coerce_template_rule(self, rule: TemplateRule | dict[str, str]) -> TemplateRule:
        if isinstance(rule, TemplateRule):
            return rule
        return TemplateRule(
            pattern=rule["pattern"],
            replacement=rule["replacement"],
            label=rule.get("label") or f"{rule['pattern']}->{rule['replacement']}",
        )

    def _mode_config(self, mode: str, target_ratio: float | None) -> CompressionMode:
        if mode not in self.MODES:
            raise ValueError(f"unknown mode: {mode}. expected one of: {', '.join(self.MODES)}")
        base = self.MODES[mode]
        if target_ratio is None:
            return base
        if not 0.2 <= target_ratio <= 1.0:
            raise ValueError("target_ratio must be between 0.2 and 1.0")
        return CompressionMode(base.name, target_ratio, base.min_anchor_recall, base.drop_optional_clauses)

    def _generate_candidates(
        self,
        original: str,
        config: CompressionMode,
        anchors: tuple[str, ...],
    ) -> list[_Candidate]:
        candidates = [_Candidate(original, (), "original")]

        templated, template_removed = self._apply_templates(original)
        candidates.append(_Candidate(templated, tuple(template_removed), "template"))

        replaced, replaced_removed = self._replace_phrases(original)
        candidates.append(_Candidate(replaced, tuple(replaced_removed), "replace"))

        template_replaced, template_replaced_removed = self._replace_phrases(templated)
        candidates.append(
            _Candidate(template_replaced, tuple(template_removed + template_replaced_removed), "template+replace")
        )

        dropped, drop_removed = self._drop_phrases(templated)
        dropped_replaced, dropped_replaced_removed = self._replace_phrases(dropped)
        candidates.append(
            _Candidate(dropped_replaced, tuple(template_removed + drop_removed + dropped_replaced_removed), "drop+replace")
        )

        trimmed = self._trim_particles(dropped_replaced)
        candidates.append(_Candidate(trimmed, tuple(template_removed + drop_removed + dropped_replaced_removed), "trim"))

        compact, compact_removed = self._drop_weak_tokens(trimmed, anchors, target_ratio=config.target_ratio)
        candidates.append(
            _Candidate(
                compact,
                tuple(template_removed + drop_removed + dropped_replaced_removed + compact_removed),
                "weak-token",
            )
        )

        if config.drop_optional_clauses:
            clause_text, clause_removed = self._drop_optional_clauses(trimmed, anchors)
            clause_compact, weak_removed = self._drop_weak_tokens(clause_text, anchors, target_ratio=config.target_ratio)
            candidates.append(
                _Candidate(
                    clause_compact,
                    tuple(template_removed + drop_removed + dropped_replaced_removed + clause_removed + weak_removed),
                    "clause+weak-token",
                )
            )

        return self._unique_candidates(candidates)

    def _apply_templates(self, text: str) -> tuple[str, list[str]]:
        candidate = text
        removed: list[str] = []

        for rule in self.template_rules:
            updated = re.sub(rule.pattern, rule.replacement, candidate)
            if updated != candidate:
                candidate = updated
                removed.append(rule.label)

        updated = re.sub(r"用户想要([^则那么。！？；;]{1,32}?)可以使用([^。！？；;]{1,32})", r"用户可用\2做\1", candidate)
        if updated != candidate:
            candidate = updated
            removed.append("用户想要A可以使用B->用户可用B做A")
        return candidate, removed

    def _select_candidate(
        self,
        original: str,
        candidates: list[_Candidate],
        config: CompressionMode,
        anchors: tuple[str, ...],
    ) -> _Candidate:
        original_len = max(1, self.count_tokens(original))
        scored: list[tuple[float, _Candidate]] = []

        for candidate in candidates:
            text = self._cleanup(candidate.text)
            if not text:
                continue
            recall = self._anchor_recall(text, anchors)
            ratio = self.count_tokens(text) / original_len
            target_distance = abs(ratio - config.target_ratio)
            too_short_penalty = 0.20 if ratio < config.target_ratio * 0.72 else 0.0
            recall_penalty = max(0.0, config.min_anchor_recall - recall) * 2.4
            score = (1.0 - ratio) + recall * 0.85 - target_distance * 0.25 - too_short_penalty - recall_penalty
            if recall >= config.min_anchor_recall or ratio >= 0.85:
                scored.append((score, _Candidate(text, candidate.removed, candidate.stage)))

        if not scored:
            return _Candidate(original, (), "fallback")

        scored.sort(key=lambda item: (item[0], -self.count_tokens(item[1].text)), reverse=True)
        return scored[0][1]

    def _drop_phrases(self, text: str) -> tuple[str, list[str]]:
        removed: list[str] = []
        candidate = text
        for phrase in sorted(self.drop_phrases, key=len, reverse=True):
            if phrase in candidate and phrase not in self.keep_words:
                candidate = candidate.replace(phrase, "")
                removed.append(phrase)
        return candidate, removed

    def _replace_phrases(self, text: str) -> tuple[str, list[str]]:
        removed: list[str] = []
        candidate = text
        for source, target in sorted(self.replacements.items(), key=lambda item: len(item[0]), reverse=True):
            if source in self.keep_words:
                continue
            if source in candidate:
                candidate = candidate.replace(source, target)
                if source != target:
                    removed.append(f"{source}->{target}")
        return candidate, removed

    def _drop_optional_clauses(self, text: str, anchors: tuple[str, ...]) -> tuple[str, list[str]]:
        clauses = re.split(r"([，；、,;])", text)
        if len(clauses) <= 1:
            return text, []

        kept: list[str] = []
        removed: list[str] = []
        for index in range(0, len(clauses), 2):
            clause = clauses[index]
            separator = clauses[index + 1] if index + 1 < len(clauses) else ""
            has_anchor = any(anchor in clause for anchor in anchors)
            is_optional = any(marker in clause for marker in self.OPTIONAL_CLAUSE_MARKERS)
            if is_optional and not has_anchor:
                removed.append(clause)
                continue
            kept.append(clause + separator)

        return "".join(kept), removed

    def _drop_weak_tokens(
        self,
        text: str,
        anchors: tuple[str, ...],
        target_ratio: float,
    ) -> tuple[str, list[str]]:
        tokens = self._segment(text)
        target_len = max(1, math.ceil(len(tokens) * target_ratio))
        if len(tokens) <= target_len:
            return text, []

        removed: list[str] = []
        kept: list[str] = []
        protected_positions = self._protected_token_positions(tokens, anchors)

        for index, token in enumerate(tokens):
            remaining = len(tokens) - index
            needed = target_len - len(kept)
            if needed >= remaining:
                kept.append(token)
                continue
            if index not in protected_positions and token in self.WEAK_TOKENS and token not in self.keep_words:
                removed.append(token)
                continue
            kept.append(token)

        return "".join(kept), removed

    def _protected_token_positions(self, tokens: list[str], anchors: tuple[str, ...]) -> set[int]:
        protected: set[int] = set()
        token_text = "".join(tokens)
        for anchor in anchors:
            start = token_text.find(anchor)
            if start < 0:
                continue
            offset = 0
            for index, token in enumerate(tokens):
                next_offset = offset + len(token)
                if start < next_offset and offset < start + len(anchor):
                    protected.add(index)
                offset = next_offset
        return protected

    def _extract_anchors(self, text: str, keywords: Iterable[str] | None = None) -> tuple[str, ...]:
        anchors: set[str] = set()
        keyword_list = tuple(keywords or ())

        for item in self.keep_words + self.domain_terms + keyword_list:
            if item and item in text:
                anchors.add(item)

        for match in re.findall(r"[A-Za-z][A-Za-z0-9_+-]*|\d+(?:\.\d+)?%?", text):
            anchors.add(match)

        return tuple(sorted(anchors, key=lambda item: (-len(item), item)))

    def _anchor_recall(self, text: str, anchors: tuple[str, ...]) -> float:
        if not anchors:
            return 1.0
        kept = sum(1 for anchor in anchors if self._anchor_preserved(anchor, text))
        return kept / len(anchors)

    def _anchor_preserved(self, anchor: str, text: str) -> bool:
        if anchor in text:
            return True
        replacement = self.replacements.get(anchor)
        return bool(replacement and replacement in text)

    def _build_diff(self, original: str, compressed: str) -> tuple[DiffOperation, ...]:
        source_tokens = self._segment(original)
        target_tokens = self._segment(compressed)
        matcher = difflib.SequenceMatcher(a=source_tokens, b=target_tokens, autojunk=False)
        operations: list[DiffOperation] = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            source = "".join(source_tokens[i1:i2])
            target = "".join(target_tokens[j1:j2])
            operations.append(DiffOperation(op=tag, source=source, target=target))
        return tuple(operations)

    def _build_rule_diff(self, removed: tuple[str, ...]) -> tuple[DiffOperation, ...]:
        operations: list[DiffOperation] = []
        for item in removed:
            if "->" in item:
                source, target = item.split("->", 1)
                operations.append(DiffOperation("replace" if target else "delete", source, target))
            elif item:
                operations.append(DiffOperation("delete", item))
        return tuple(operations)

    def _warnings(
        self,
        original: str,
        compressed: str,
        anchors: tuple[str, ...],
        recall: float,
        config: CompressionMode,
    ) -> tuple[str, ...]:
        warnings: list[str] = []
        if recall < config.min_anchor_recall:
            warnings.append(f"anchor_recall_below_{config.min_anchor_recall}")
        for word in ("不", "没有", "无法", "不能", "禁止", "未"):
            if word in original and word not in compressed:
                warnings.append(f"lost_negation:{word}")
        for number in re.findall(r"\d+(?:\.\d+)?%?", original):
            if number not in compressed:
                warnings.append(f"lost_number:{number}")
        for anchor in anchors:
            if anchor in self.keep_words and not self._anchor_preserved(anchor, compressed):
                warnings.append(f"lost_keep_word:{anchor}")
        return tuple(dict.fromkeys(warnings))

    def _unique_candidates(self, candidates: list[_Candidate]) -> list[_Candidate]:
        seen: set[str] = set()
        unique: list[_Candidate] = []
        for candidate in candidates:
            text = self._cleanup(candidate.text)
            if text and text not in seen:
                unique.append(_Candidate(text, candidate.removed, candidate.stage))
                seen.add(text)
        return unique

    def _dedupe(self, text: str) -> str:
        text = re.sub(rf"([{re.escape(self.PUNCTUATION)}])\1+", r"\1", text)
        text = re.sub(r"(.{1,4})\1+", r"\1", text)
        return text

    def _trim_particles(self, text: str) -> str:
        text = re.sub(rf"[啊呀呢吧嘛啦喔哦]+(?=[{re.escape(self.PUNCTUATION)}]|$)", "", text)
        text = re.sub(rf"的(?=[{re.escape(self.PUNCTUATION)}]|$)", "", text)
        return text

    def _cleanup(self, text: str) -> str:
        text = self._dedupe(text)
        text = re.sub(r"\s+", "", text)
        text = re.sub(rf"^[{re.escape(self.PUNCTUATION)}]+", "", text)
        text = re.sub(rf"[，；、,;]+$", "", text)
        text = re.sub(rf"([{re.escape(self.PUNCTUATION)}])+", lambda match: match.group(0)[0], text)
        return text.strip()

    def _split_sentences(self, text: str) -> list[str]:
        parts = re.findall(r"[^。！？!?；;]+[。！？!?；;]?", text)
        return [part.strip() for part in parts if part.strip()]

    def _sentence_signature(self, sentence: str) -> str:
        normalized = self._normalize(sentence)
        normalized = re.sub(rf"[{re.escape(self.PUNCTUATION)}]", "", normalized)
        for phrase in self.drop_phrases:
            normalized = normalized.replace(phrase, "")
        normalized, _ = self._replace_phrases(normalized)
        return normalized

    def _cleanup_paragraph(self, text: str) -> str:
        text = re.sub(r"\s+", "", text.strip())
        text = re.sub(r"([。！？!?；;])+", lambda match: match.group(0)[0], text)
        return text

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", "", text.strip())

    def _segment(self, text: str) -> list[str]:
        tokens: list[str] = []
        index = 0
        while index < len(text):
            ascii_match = re.match(r"[A-Za-z0-9_+-]+", text[index:])
            if ascii_match:
                tokens.append(ascii_match.group(0))
                index += len(ascii_match.group(0))
                continue

            matched = None
            for word in self.lexicon:
                if text.startswith(word, index):
                    matched = word
                    break
            if matched:
                tokens.append(matched)
                index += len(matched)
                continue

            char = text[index]
            if not char.isspace():
                tokens.append(char)
            index += 1
        return tokens
