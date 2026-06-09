# zh-token-compressor

一个轻量中文 token 压缩器：输入一句话，输出更短且尽量保留等效语义的句子。

当前版本不是大语言模型，而是一个可解释的混合压缩器：

- 生成多个压缩候选
- 按压缩率、关键词保留率和目标比例打分
- 保护否定词、数字、英文 token 和指定关键词
- 可选使用 `tiktoken` 统计真实 LLM token
- 支持 `safe`、`balanced`、`aggressive` 三种压缩模式
- 支持段落级压缩：按句压缩、删除重复句、汇总风险警告
- 支持结构模板：`如果 A 那么 B -> 若 A 则 B`、`由于 A 所以 B -> 因 A 故 B`
- 支持从 `原句 -> 压缩句` CSV 中学习删词和替换规则
- 支持批量压缩、平均评估和 benchmark 回归测试

## 快速使用

```powershell
cd D:\Gititem\token-compressor
D:\uvenv\Scripts\python.exe -m token_compressor.cli "我认为这个功能其实能够帮助用户非常快速地完成文本压缩"
```

输出示例：

```text
功能助用户快速做完压缩文本
tokens: 22 -> 9, ratio=0.409, mode=balanced, anchor_recall=1.0
```

如果 `python` 已加入 PATH，也可以使用：

```powershell
python -m token_compressor.cli "我认为这个功能其实能够帮助用户非常快速地完成文本压缩"
```

## 压缩模式

```powershell
python -m token_compressor.cli "我认为这个工具其实能够帮助开发者非常快速地减少 token 使用量并且提高提示词表达效率" --mode safe
python -m token_compressor.cli "我认为这个工具其实能够帮助开发者非常快速地减少 token 使用量并且提高提示词表达效率" --mode balanced
python -m token_compressor.cli "我认为这个工具其实能够帮助开发者非常快速地减少 token 使用量并且提高提示词表达效率" --mode aggressive
```

模式说明：

- `safe`：更保守，优先保留语义，适合高风险文本
- `balanced`：默认模式，兼顾压缩率和可读性
- `aggressive`：更短，但可能牺牲一些细节

也可以手动指定目标比例：

```powershell
python -m token_compressor.cli "由于现在输入内容比较长所以我们需要进行压缩处理" --ratio 0.55
```

## 关键词保护

```powershell
python -m token_compressor.cli "这个模型不能删除 30% 这个数字，也必须保留 API 关键词" --mode aggressive --keyword API --details
```

压缩器会尽量保留：

- 否定词：`不`、`不能`、`没有`、`无法`、`未`
- 数字和百分比：`30%`
- 英文 token：`API`、`token`
- 通过 `--keyword` 指定的关键词

## JSON 输出

```powershell
python -m token_compressor.cli "如果用户想要减少 token 使用量那么可以使用这个压缩模型" --json
```

## Token 计数器

默认 `--token-counter auto` 会优先尝试 `tiktoken`，如果本地没有安装，会自动回退到内置粗计数器。

强制使用内置计数：

```powershell
python -m token_compressor.cli "如果用户想要减少 token 使用量那么可以使用这个压缩模型" --token-counter coarse --details
```

安装并强制使用 `tiktoken`：

```powershell
pip install ".[tiktoken]"
python -m token_compressor.cli "如果用户想要减少 token 使用量那么可以使用这个压缩模型" --token-counter tiktoken --details
```

## 批量压缩

准备一个 UTF-8 文本文件，每行一句：

```powershell
python -m token_compressor.cli --input-file examples\sentences.txt --mode balanced --json
```

## 段落级压缩

段落压缩会先切句，再压缩每句，并删除近似重复句：

```powershell
python -m token_compressor.cli --input-file examples\paragraph.txt --paragraph --details --token-counter coarse
```

也可以直接传入一整段：

```powershell
python -m token_compressor.cli "我认为这个功能其实能够帮助用户非常快速地完成文本压缩。如果用户想要减少 token 使用量那么可以使用这个压缩模型。" --paragraph --details
```

## 模板规则

除了短语替换，压缩器还会生成结构模板候选：

```text
如果 A 那么 B -> 若 A 则 B
由于 A 所以 B -> 因 A 故 B
因为 A 所以 B -> 因 A 故 B
用户想要 A 可以使用 B -> 用户可用 B 做 A
```

示例：

```powershell
python -m token_compressor.cli "由于现在输入内容比较长所以我们需要进行压缩处理" --details --token-counter coarse
```

## 评估平均压缩效果

```powershell
python -m token_compressor.cli --evaluate examples\sentences.txt --mode balanced
```

输出包含：

- `avg_ratio`：平均压缩后 token 比例
- `avg_anchor_recall`：关键词平均保留率
- `warning_rate`：出现语义风险警告的比例

## Benchmark 回归测试

`examples\benchmark.csv` 可以为每条样本设置模式、必须保留关键词、最大压缩比例和最低关键词保留率。

```powershell
python -m token_compressor.cli --benchmark examples\benchmark.csv --token-counter coarse
```

在 CI 或提交前检查时，可以让 benchmark 失败返回非零退出码：

```powershell
python -m token_compressor.cli --benchmark examples\benchmark.csv --fail-on-benchmark
```

## 从样本学习规则

准备 CSV，包含两列：`original` 和 `compressed`。

```powershell
python -m token_compressor.cli --learn examples\train_pairs.csv --save-profile profile.json
python -m token_compressor.cli "如果用户想要减少 token 使用量那么可以使用这个压缩模型" --profile examples\profile.json
```

## Python 调用

```python
from token_compressor import TokenCompressor

compressor = TokenCompressor()
result = compressor.compress(
    "我认为这个功能其实能够帮助用户非常快速地完成文本压缩",
    mode="balanced",
)
print(result.compressed)
print(result.compression_ratio, result.anchor_recall)
```

批量评估：

```python
metrics = compressor.evaluate([
    "我认为这个功能其实能够帮助用户非常快速地完成文本压缩",
    "如果用户想要减少 token 使用量那么可以使用这个压缩模型",
])
print(metrics)
```

Benchmark：

```python
from token_compressor.benchmark import load_benchmark, run_benchmark

cases = load_benchmark("examples/benchmark.csv")
report = run_benchmark(compressor, cases)
print(report.to_dict())
```

## 测试

```powershell
python -m unittest discover -s tests -v
```

## 边界

这个项目仍然是轻量规则和打分模型，不等同于真正的神经网络语义压缩模型。它适合 prompt 预处理、去口水词、构造训练 baseline。如果要进一步做到更稳定的抽象改写，需要用大量 `长句 -> 短句` 样本训练 T5/BART/mT5 类 seq2seq 模型，或接入 LLM 做重写。
