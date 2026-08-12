import json
import tempfile
import unittest
from pathlib import Path

from verifierfuzz.dataset import DatasetColumns, load_dataset_cases


class DatasetTests(unittest.TestCase):
    def test_jsonl_supports_nested_column_paths_and_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rollouts.jsonl"
            rows = [
                {
                    "uid": "first",
                    "prompt": "q1",
                    "rollout": {"text": "a1"},
                    "reward_model": {"ground_truth": "a1"},
                    "extra_info": {"split": "train"},
                },
                {
                    "uid": "second",
                    "prompt": "q2",
                    "rollout": {"text": "a2"},
                    "reward_model": {"ground_truth": "a2"},
                    "extra_info": {"split": "test"},
                },
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            cases = load_dataset_cases(
                str(path),
                columns=DatasetColumns(
                    prompt="prompt",
                    completion="rollout.text",
                    reference="reward_model.ground_truth",
                    case_id="uid",
                    metadata=["extra_info.split"],
                ),
                framework="verl",
                offset=1,
                limit=1,
            )

            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0].case_id, "second")
            self.assertEqual(cases[0].completion, "a2")
            self.assertEqual(cases[0].reference, "a2")
            self.assertEqual(cases[0].metadata["framework"], "verl")
            self.assertEqual(cases[0].metadata["extra_info.split"], "test")

    def test_json_array_uses_stable_row_id_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rollouts.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "prompt": "question",
                            "response": "answer",
                            "ground_truth": "answer",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            cases = load_dataset_cases(str(path))

            self.assertEqual(cases[0].case_id, "row-0")

    def test_missing_completion_fails_with_column_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rollouts.jsonl"
            path.write_text('{"prompt":"question"}\n', encoding="utf-8")

            with self.assertRaisesRegex(KeyError, "response"):
                load_dataset_cases(str(path))


if __name__ == "__main__":
    unittest.main()
