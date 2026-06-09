import unittest

from token_compressor import TokenCompressor


class TokenCompressorTest(unittest.TestCase):
    def test_compress_shortens_sentence(self) -> None:
        compressor = TokenCompressor()
        result = compressor.compress("我认为这个功能其实能够帮助用户非常快速地完成文本压缩")

        self.assertTrue(result.compressed)
        self.assertNotEqual(result.compressed, result.original)
        self.assertLess(result.compressed_tokens, result.original_tokens)
        self.assertIn("用户", result.compressed)
        self.assertIn("压缩", result.compressed)

    def test_learn_profile_from_pairs(self) -> None:
        compressor = TokenCompressor()
        profile = compressor.learn_profile(
            [
                ("我认为这个功能其实可以帮助用户完成文本压缩", "功能可帮助用户压缩文本"),
                ("我认为这个工具其实可以帮助开发者完成提示词压缩", "工具可帮助开发者压缩提示词"),
            ]
        )

        self.assertTrue(profile["drop_phrases"])


if __name__ == "__main__":
    unittest.main()
