import unittest


class CorpusPackageImportTests(unittest.TestCase):
    def test_gate_run_importable(self):
        from corpus.index import gate_run

        self.assertTrue(callable(gate_run))


if __name__ == "__main__":
    unittest.main()
