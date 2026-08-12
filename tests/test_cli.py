import json
import tempfile
import unittest
from pathlib import Path

from verifierfuzz.__main__ import main


MODULE = """
def target(completion, reference):
    return "42" in completion

def oracle(completion, reference):
    return completion.strip() == reference

async def slime_target(args, sample):
    return "42" in sample.response

async def slime_oracle(args, sample):
    return sample.response.strip() == sample.label
"""


class CliTests(unittest.TestCase):
    def test_audit_writes_jsonl_and_sarif(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            module = root / "rewards.py"
            corpus = root / "cases.jsonl"
            output = root / "findings.jsonl"
            sarif = root / "findings.sarif"
            module.write_text(MODULE, encoding="utf-8")
            corpus.write_text(
                json.dumps(
                    {
                        "case_id": "math",
                        "completion": r"\boxed{42}",
                        "reference": r"\boxed{42}",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            exit_code = main(
                [
                    "audit",
                    "--target",
                    f"{module}:target",
                    "--oracle",
                    f"{module}:oracle",
                    "--adapter",
                    "pair",
                    "--oracle-adapter",
                    "pair",
                    "--target-policy",
                    "zero-one",
                    "--oracle-policy",
                    "zero-one",
                    "--corpus",
                    str(corpus),
                    "--output",
                    str(output),
                    "--sarif",
                    str(sarif),
                    "--max-findings",
                    "99",
                ]
            )

            self.assertEqual(exit_code, 0)
            finding = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(finding["schema_version"], 1)
            self.assertEqual(finding["kind"], "false_positive")
            sarif_payload = json.loads(sarif.read_text(encoding="utf-8"))
            self.assertEqual(sarif_payload["version"], "2.1.0")

    def test_regression_distinguishes_fixed_and_known_cases(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            module = root / "rewards.py"
            corpus = root / "regression.jsonl"
            module.write_text(MODULE, encoding="utf-8")
            rows = [
                {
                    "case": {
                        "case_id": "known",
                        "completion": "answer 42",
                        "reference": "43",
                    },
                    "expected_kind": "false_positive",
                },
                {
                    "case": {
                        "case_id": "fixed",
                        "completion": "42",
                        "reference": "42",
                    },
                    "expected_kind": None,
                },
            ]
            corpus.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            exit_code = main(
                [
                    "regression",
                    "--target",
                    f"{module}:target",
                    "--oracle",
                    f"{module}:oracle",
                    "--adapter",
                    "pair",
                    "--oracle-adapter",
                    "pair",
                    "--target-policy",
                    "zero-one",
                    "--oracle-policy",
                    "zero-one",
                    "--corpus",
                    str(corpus),
                ]
            )

            self.assertEqual(exit_code, 0)

    def test_replay_supports_slime_reward_adapter(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            module = root / "rewards.py"
            corpus = root / "cases.jsonl"
            module.write_text(MODULE, encoding="utf-8")
            corpus.write_text(
                json.dumps(
                    {
                        "case_id": "slime",
                        "completion": "answer 42",
                        "reference": "43",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            exit_code = main(
                [
                    "replay",
                    "--target",
                    f"{module}:slime_target",
                    "--oracle",
                    f"{module}:slime_oracle",
                    "--adapter",
                    "slime",
                    "--oracle-adapter",
                    "slime",
                    "--target-policy",
                    "zero-one",
                    "--oracle-policy",
                    "zero-one",
                    "--corpus",
                    str(corpus),
                    "--max-findings",
                    "1",
                ]
            )

            self.assertEqual(exit_code, 0)

    def test_scan_maps_dataset_columns_and_freezes_regressions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            module = root / "rewards.py"
            dataset = root / "rollouts.jsonl"
            findings = root / "findings.jsonl"
            regression = root / "regression.jsonl"
            module.write_text(MODULE, encoding="utf-8")
            dataset.write_text(
                json.dumps(
                    {
                        "uid": "scan",
                        "prompt": "question",
                        "rollout": {"text": "answer 42"},
                        "reward_model": {"ground_truth": "43"},
                        "extra_info": {"split": "train"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            exit_code = main(
                [
                    "scan",
                    "--target",
                    f"{module}:target",
                    "--oracle",
                    f"{module}:oracle",
                    "--adapter",
                    "pair",
                    "--oracle-adapter",
                    "pair",
                    "--target-policy",
                    "zero-one",
                    "--oracle-policy",
                    "zero-one",
                    "--dataset",
                    str(dataset),
                    "--id-column",
                    "uid",
                    "--completion-column",
                    "rollout.text",
                    "--reference-column",
                    "reward_model.ground_truth",
                    "--metadata-column",
                    "extra_info.split",
                    "--minimize",
                    "--output",
                    str(findings),
                    "--regression-output",
                    str(regression),
                    "--max-findings",
                    "99",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(findings.exists())
            frozen = [
                json.loads(line)
                for line in regression.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(frozen[0]["expected_kind"], "false_positive")
            self.assertEqual(
                frozen[0]["case"]["metadata"]["extra_info.split"],
                "train",
            )


if __name__ == "__main__":
    unittest.main()
