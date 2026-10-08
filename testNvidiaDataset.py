import json
import tempfile
import unittest
from pathlib import Path

from NVIDIA.dataset_utils import load_vigilancia_sanitaria_dataset


class NvidiaDatasetTests(unittest.TestCase):
    def test_load_vigilancia_sanitaria_dataset_formats_records(self):
        dataset_path = Path(__file__).parent / "data" / "raw" / "03_vigilancia_sanitaria.jsonl"

        dataset = load_vigilancia_sanitaria_dataset(dataset_path)

        self.assertEqual(len(dataset), 1805)
        self.assertTrue(all("text" in row for row in dataset))
        self.assertTrue(all("instruction" not in row for row in dataset))
        self.assertIn("### Pergunta:", dataset[0]["text"])
        self.assertIn("### Resposta:", dataset[0]["text"])

    def test_load_vigilancia_sanitaria_dataset_ignores_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.jsonl"
            path.write_text(
                "// section marker\n"
                + json.dumps({"instruction": "valid", "input": "", "output": "answer"})
                + "\n",
                encoding="utf-8",
            )

            dataset = load_vigilancia_sanitaria_dataset(path)

            self.assertEqual(len(dataset), 1)
            self.assertEqual(
                dataset[0]["text"],
                "### Pergunta: valid\n\n### Resposta: answer",
            )


if __name__ == "__main__":
    unittest.main()
