import unittest

from token_compressor.benchmark import load_benchmark, run_benchmark
from token_compressor import TokenCompressor


class FixedTokenCounter:
    name = "fixed"

    def count(self, text: str) -> int:
        return max(1, len(text) // 2)


class TokenCompressorTest(unittest.TestCase):
    def test_compress_shortens_sentence(self) -> None:
        compressor = TokenCompressor()
        result = compressor.compress("我认为这个功能其实能够帮助用户非常快速地完成文本压缩")

        self.assertTrue(result.compressed)
        self.assertNotEqual(result.compressed, result.original)
        self.assertLess(result.compressed_tokens, result.original_tokens)
        self.assertIn("用户", result.compressed)
        self.assertIn("压缩", result.compressed)
        self.assertGreaterEqual(result.anchor_recall, 0.8)

    def test_modes_have_different_strength(self) -> None:
        compressor = TokenCompressor()
        text = "我认为这个工具其实能够帮助开发者非常快速地减少 token 使用量并且提高提示词表达效率"

        safe = compressor.compress(text, mode="safe")
        aggressive = compressor.compress(text, mode="aggressive")

        self.assertLessEqual(aggressive.compressed_tokens, safe.compressed_tokens)
        self.assertEqual(safe.mode, "safe")
        self.assertEqual(aggressive.mode, "aggressive")

    def test_preserves_negation_numbers_and_keywords(self) -> None:
        compressor = TokenCompressor()
        result = compressor.compress(
            "这个模型不能删除 30% 这个数字，也必须保留 API 关键词",
            mode="aggressive",
            keywords=("API",),
        )

        self.assertIn("不能", result.compressed)
        self.assertIn("30%", result.compressed)
        self.assertIn("API", result.compressed)
        self.assertFalse(any(item.startswith("lost_") for item in result.warnings))

    def test_evaluate_returns_metrics(self) -> None:
        compressor = TokenCompressor()
        metrics = compressor.evaluate(
            [
                "我认为这个功能其实能够帮助用户非常快速地完成文本压缩",
                "如果用户想要减少 token 使用量那么可以使用这个压缩模型",
            ]
        )

        self.assertEqual(metrics["count"], 2.0)
        self.assertLess(metrics["avg_ratio"], 1.0)
        self.assertGreater(metrics["avg_anchor_recall"], 0.0)

    def test_can_force_coarse_token_counter(self) -> None:
        compressor = TokenCompressor(token_counter="coarse")
        result = compressor.compress("如果用户想要减少 token 使用量那么可以使用这个压缩模型")

        self.assertEqual(result.token_counter, "coarse")
        self.assertLess(result.compression_ratio, 1.0)

    def test_accepts_custom_token_counter(self) -> None:
        compressor = TokenCompressor(token_counter=FixedTokenCounter())
        result = compressor.compress("我认为这个功能其实能够帮助用户非常快速地完成文本压缩")

        self.assertEqual(result.token_counter, "fixed")
        self.assertGreater(result.original_tokens, 0)

    def test_benchmark_reports_failures(self) -> None:
        compressor = TokenCompressor(token_counter="coarse")
        cases = load_benchmark("examples/benchmark.csv")
        report = run_benchmark(compressor, cases)

        self.assertGreater(report.count, 0)
        self.assertEqual(report.token_counter, "coarse")
        self.assertGreaterEqual(report.passed, 1)

    def test_learn_profile_from_pairs(self) -> None:
        compressor = TokenCompressor()
        profile = compressor.learn_profile(
            [
                ("我认为这个功能其实可以帮助用户完成文本压缩", "功能可帮助用户压缩文本"),
                ("我认为这个工具其实可以帮助开发者完成提示词压缩", "工具可帮助开发者压缩提示词"),
            ]
        )

        self.assertTrue(profile["drop_phrases"])
        self.assertIn("domain_terms", profile)


if __name__ == "__main__":
    unittest.main()
