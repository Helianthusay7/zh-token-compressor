from __future__ import annotations

import json
from importlib import resources
from typing import Iterable

from .compressor import TokenCompressor, TokenCounter


DOMAIN_NAMES = (
    "general",
    "business",
    "tech",
    "product",
    "report",
    "legal",
    "finance",
    "education",
    "medical",
)


DOMAIN_KEYWORDS = {
    "business": ("业务", "流程", "线上化", "落地", "可视化", "项目", "模块", "管理", "效率"),
    "tech": ("API", "接口", "数据库", "服务", "延迟", "吞吐", "QPS", "错误码", "缓存", "部署"),
    "product": ("需求", "用户", "体验", "功能", "版本", "迭代", "原型", "场景", "交互"),
    "report": ("本阶段", "推进", "完成", "风险", "进展", "计划", "问题", "复盘", "汇报"),
    "legal": ("甲方", "乙方", "协议", "合同", "违约", "责任", "期限", "赔偿", "条款"),
    "finance": ("收入", "成本", "利润", "预算", "同比", "环比", "现金流", "风险", "资产"),
    "education": ("课程", "学生", "教学", "学习", "培训", "作业", "考试", "知识点", "能力"),
    "medical": ("患者", "症状", "诊断", "剂量", "用药", "禁忌", "治疗", "检查", "指标"),
}


def detect_domain(text: str) -> str:
    scores = {
        domain: sum(1 for keyword in keywords if keyword and keyword in text)
        for domain, keywords in DOMAIN_KEYWORDS.items()
    }
    best_domain, best_score = max(scores.items(), key=lambda item: item[1])
    return best_domain if best_score > 0 else "general"


def load_domain_config(domain: str) -> dict[str, object]:
    if domain not in DOMAIN_NAMES:
        raise ValueError(f"unknown domain: {domain}")
    path = resources.files("token_compressor.domain_configs").joinpath(f"{domain}.json")
    return json.loads(path.read_text(encoding="utf-8"))


def build_domain_compressor(
    domain: str,
    sample_text: str = "",
    token_counter: str | TokenCounter = "auto",
    tiktoken_encoding: str = "cl100k_base",
) -> tuple[str, TokenCompressor]:
    selected = detect_domain(sample_text) if domain == "auto" else domain
    config = load_domain_config(selected)
    return selected, TokenCompressor.from_config_data(
        config,
        token_counter=token_counter,
        tiktoken_encoding=tiktoken_encoding,
    )


def domain_choices() -> Iterable[str]:
    return ("auto",) + DOMAIN_NAMES
