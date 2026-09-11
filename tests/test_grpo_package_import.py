import unittest


class GrpoPackageImportTests(unittest.TestCase):
    def test_train_module_importable(self):
        from training.grpo.train import build_parser

        self.assertTrue(callable(build_parser))


if __name__ == "__main__":
    unittest.main()
