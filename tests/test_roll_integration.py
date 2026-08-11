import unittest
from types import SimpleNamespace

from verifierfuzz.integrations import (
    CallableVerifier,
    RollVerifier,
    ShadowAuditor,
    make_roll_cases,
    wrap_roll_compute_rewards,
)
from verifierfuzz.protocol import Decision, ScorePolicy, VerifierCase


class FakeDataProto:
    def __init__(self, batch, non_tensor_batch=None, meta_info=None):
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch or {}
        self.meta_info = meta_info or {}

    def __len__(self):
        return len(next(iter(self.batch.values())))


class FakeTokenizer:
    def batch_decode(self, rows, skip_special_tokens=False):
        prefix = "clean" if skip_special_tokens else "raw"
        return [f"{prefix}:{','.join(str(token) for token in row)}" for row in rows]


class RollIntegrationTests(unittest.TestCase):
    def test_make_cases_decodes_official_dataproto_fields(self):
        data = FakeDataProto(
            batch={
                "prompts": [[1, 2], [3]],
                "responses": [[4], [5, 6]],
            },
            non_tensor_batch={
                "id": ["roll-1", "roll-2"],
                "ground_truth": ["4", "5"],
                "tag": ["math", "math"],
            },
            meta_info={"global_step": 8},
        )

        cases = make_roll_cases(data, FakeTokenizer(), indices=[1])

        self.assertEqual(cases[0].case_id, "roll-2")
        self.assertEqual(cases[0].prompt, "raw:3")
        self.assertEqual(cases[0].completion, "raw:5,6")
        self.assertEqual(cases[0].metadata["roll_fields"]["tag"], "math")
        self.assertEqual(cases[0].metadata["meta_info"]["global_step"], 8)

    def test_wrapper_returns_exact_original_dataproto(self):
        data = FakeDataProto(
            batch={"prompts": [[1]], "responses": [[2]]},
            non_tensor_batch={
                "id": ["roll"],
                "ground_truth": ["right"],
            },
        )
        output = FakeDataProto(
            batch={
                "token_level_rewards": [[0.0]],
                "response_level_rewards": [1.0],
                "scores": [1.0],
            }
        )
        findings = []
        auditor = ShadowAuditor(
            CallableVerifier(
                lambda case: False,
                policy=ScorePolicy.zero_one(),
            ),
            target_policy=ScorePolicy.zero_one(),
            sink=findings.append,
        )
        wrapped = wrap_roll_compute_rewards(
            lambda input_data: output,
            auditor,
            FakeTokenizer(),
            sample_rate=1.0,
        )

        result = wrapped(data)
        auditor.close()

        self.assertIs(result, output)
        self.assertEqual(findings[0].kind, "false_positive")
        self.assertEqual(findings[0].case.case_id, "roll")

    def test_zero_sampling_does_not_require_tokenizer(self):
        data = FakeDataProto(
            batch={"prompts": [[1]], "responses": [[2]]},
            non_tensor_batch={"ground_truth": ["answer"]},
        )
        output = FakeDataProto(batch={"response_level_rewards": [0.0]})
        auditor = ShadowAuditor(CallableVerifier(lambda case: True))
        wrapped = wrap_roll_compute_rewards(
            lambda input_data: output,
            auditor,
            sample_rate=0.0,
        )

        self.assertIs(wrapped(data), output)
        auditor.close()
        self.assertEqual(auditor.stats().submitted, 0)

    def test_decode_failure_is_fail_open(self):
        data = FakeDataProto(
            batch={"prompts": [[1]], "responses": [[2]]},
            non_tensor_batch={"ground_truth": ["answer"]},
        )
        output = FakeDataProto(batch={"response_level_rewards": [1.0]})
        auditor = ShadowAuditor(CallableVerifier(lambda case: True))
        wrapped = wrap_roll_compute_rewards(
            lambda input_data: output,
            auditor,
            sample_rate=1.0,
        )

        self.assertIs(wrapped(data), output)
        auditor.close()
        self.assertEqual(auditor.stats().errors, 1)

    def test_roll_verifier_runs_worker_compute_rewards(self):
        output = FakeDataProto(batch={"response_level_rewards": [1.0]})

        class Worker:
            def compute_rewards(self, data):
                return output

        verifier = RollVerifier(
            Worker(),
            lambda case: SimpleNamespace(case=case),
            policy=ScorePolicy.zero_one(),
        )

        import asyncio

        outcome = asyncio.run(
            verifier.evaluate(VerifierCase(case_id="offline", completion="answer"))
        )
        self.assertEqual(outcome.decision, Decision.PASS)


if __name__ == "__main__":
    unittest.main()
