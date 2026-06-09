# token-compressor

一个轻量中文 token 压缩模型原型：输入一句话，输出更短且尽量保留等效语义的句子。

它不是大语言模型，而是可解释的规则压缩器，适合做：

- prompt 预压缩
- 数据集清洗
- “原句 -> 短句”训练数据的 baseline
- 后续替换为 Transformer/Seq2Seq 模型前的最小可用版本

## 使用

```powershell
cd D:\Gititem\token-compressor
python -m token_compressor.cli "我认为这个功能其实能够帮助用户非常快速地完成文本压缩"
```

如果 `python` 没有加入 PATH，可以使用本机已有解释器：

```powershell
D:\uvenv\Scripts\python.exe -m token_compressor.cli "我认为这个功能其实能够帮助用户非常快速地完成文本压缩"
```

输出示例：

```text
功能能助用户快速做完文本压缩
tokens: 28 -> 15, ratio=0.536
removed: 我认为 / 其实 / 非常
```

指定压缩强度：

```powershell
python -m token_compressor.cli "由于现在输入内容比较长所以我们需要进行压缩处理" --ratio 0.55
```

## 从样本学习删词规则

准备 CSV，包含两列：`original` 和 `compressed`。

```powershell
python -m token_compressor.cli --learn examples\train_pairs.csv --save-profile profile.json
python -m token_compressor.cli "如果用户想要减少 token 使用量那么可以使用这个压缩模型" --profile examples\profile.json
```

## 测试

```powershell
python -m unittest discover -s tests -v
```

## Python 调用

```python
from token_compressor import TokenCompressor

compressor = TokenCompressor()
result = compressor.compress("我认为这个功能其实能够帮助用户非常快速地完成文本压缩")
print(result.compressed)
```

## 设计说明

当前实现包括：

- 中文粗 token 计数
- 删除低信息短语，例如“我认为”“其实”“非常”
- 等价短语替换，例如“因为 -> 因”“如果 -> 若”
- 删除部分语气词和重复表达
- 从成对样本中学习常被删除的词

如果你后续需要真正的神经网络版本，可以把这里的 `examples/train_pairs.csv` 扩展为大规模训练集，再用 T5/BART/mT5 训练“长句到短句”的生成模型。
