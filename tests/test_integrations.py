import asyncio
import threading
import unittest

from verifierfuzz.integrations import (
    CallableVerifier,
    ShadowAuditor,
    TrlVerifier,
    VerlVerifier,
    wrap_trl_reward,
    wrap_verl_reward,
)
from verifierfuzz.protocol import (
    Decision,
    ScorePolicy,
    VerifierCase,
    VerifierOutcome,
)


class IntegrationTests(unittest.TestCase):
    def test_verl_adapter_uses_official_custom_reward_signature(self):
        calls = []

        def reward_fn(data_source, solution_str, ground_truth, extra_info=None):
            calls.append((data_source, solution_str, ground_truth, extra_info))
            return 1 if solution_str == ground_truth else -1

        verifier = VerlVerifier(reward_fn, policy=ScorePolicy.signed())
        outcome = asyncio.run(
            verifier.evaluate(
                VerifierCase(
                    case_id="v",
                    completion="42",
                    reference="42",
                    metadata={
                        "data_source": "math",
                        "extra_info": {"uid": "v"},
                    },
                )
            )
        )

        self.assertEqual(outcome.decision, Decision.PASS)
        self.assertEqual(calls[0], ("math", "42", "42", {"uid": "v"}))

    def test_trl_adapter_uses_batch_signature(self):
        calls = []

        def reward_fn(prompts, completions, **kwargs):
            calls.append((prompts, completions, kwargs))
            return [1.0]

        verifier = TrlVerifier(reward_fn, policy=ScorePolicy.zero_one())
        outcome = asyncio.run(
            verifier.evaluate(
                VerifierCase(
                    case_id="t",
                    prompt="question",
                    completion="answer",
                    reference="answer",
                    metadata={"dataset_columns": {"ground_truth": ["answer"]}},
                )
            )
        )

        self.assertEqual(outcome.decision, Decision.PASS)
        self.assertEqual(calls[0][0], ["question"])
        self.assertEqual(calls[0][1], ["answer"])

    def test_verl_wrapper_returns_exact_original_reward(self):
        raw_reward = {"score": 1.0, "msg": "original"}

        def reward_fn(data_source, solution_str, ground_truth, extra_info=None):
            return raw_reward

        findings = []
        reference = CallableVerifier(
            lambda case: False,
            policy=ScorePolicy.zero_one(),
        )
        auditor = ShadowAuditor(
            reference,
            target_policy=ScorePolicy.zero_one(),
            sink=findings.append,
        )
        wrapped = wrap_verl_reward(reward_fn, auditor, sample_rate=1.0)

        result = wrapped("math", "wrong 42", "43", {"uid": "one"})
        auditor.close()

        self.assertIs(result, raw_reward)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].kind, "false_positive")

    def test_trl_wrapper_returns_exact_original_batch(self):
        raw_rewards = [1.0, 0.0]

        def reward_fn(prompts, completions, **kwargs):
            return raw_rewards

        reference = CallableVerifier(
            lambda case: case.completion == case.reference,
            policy=ScorePolicy.zero_one(),
        )
        auditor = ShadowAuditor(reference, target_policy=ScorePolicy.zero_one())
        wrapped = wrap_trl_reward(reward_fn, auditor, sample_rate=1.0)

        result = wrapped(
            prompts=["q1", "q2"],
            completions=["a1", "bad"],
            ground_truth=["a1", "a2"],
        )
        auditor.close()

        self.assertIs(result, raw_rewards)
        self.assertEqual(auditor.stats().processed, 2)

    def test_shadow_failure_does_not_change_training_reward(self):
        def reward_fn(data_source, solution_str, ground_truth, extra_info=None):
            return 0.25

        class BrokenReference:
            def evaluate(self, case):
                raise RuntimeError("oracle unavailable")

        auditor = ShadowAuditor(BrokenReference())
        wrapped = wrap_verl_reward(reward_fn, auditor, sample_rate=1.0)

        result = wrapped("math", "answer", "answer")
        auditor.close()

        self.assertEqual(result, 0.25)
        self.assertEqual(auditor.stats().processed, 1)

    def test_async_reference_timeout_is_reported_off_path(self):
        class SlowReference:
            async def evaluate(self, case):
                await asyncio.sleep(1)
                return VerifierOutcome(decision=Decision.PASS, score=1)

        findings = []
        auditor = ShadowAuditor(
            SlowReference(),
            sink=findings.append,
            reference_timeout=0.01,
        )
        wrapped = wrap_verl_reward(
            lambda data_source, solution_str, ground_truth, extra_info=None: 1.0,
            auditor,
            sample_rate=1.0,
        )

        result = wrapped("math", "answer", "answer")
        auditor.close()

        self.assertEqual(result, 1.0)
        self.assertEqual(findings[0].kind, "reference_error")

    def test_shadow_queue_drops_instead_of_blocking(self):
        started = threading.Event()
        release = threading.Event()

        class BlockingReference:
            def evaluate(self, case):
                started.set()
                release.wait(timeout=2)
                return VerifierOutcome(decision=Decision.FAIL, score=0)

        auditor = ShadowAuditor(
            BlockingReference(),
            target_policy=ScorePolicy.zero_one(),
            max_queue_size=1,
        )
        case = VerifierCase(case_id="q", completion="x")
        self.assertTrue(auditor.try_submit(case, 1))
        self.assertTrue(started.wait(timeout=1))
        self.assertTrue(auditor.try_submit(case, 1))
        self.assertFalse(auditor.try_submit(case, 1))
        release.set()
        auditor.close()

        self.assertGreaterEqual(auditor.stats().dropped, 1)


class AsyncIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_verl_wrapper_preserves_reward(self):
        async def reward_fn(data_source, solution_str, ground_truth, extra_info=None):
            return 0.75

        reference = CallableVerifier(
            lambda case: True,
            policy=ScorePolicy.zero_one(),
        )
        auditor = ShadowAuditor(reference)
        wrapped = wrap_verl_reward(reward_fn, auditor, sample_rate=0)

        result = await wrapped("math", "answer", "answer")
        auditor.close()

        self.assertEqual(result, 0.75)

    async def test_async_trl_wrapper_preserves_batch(self):
        rewards = [1.0]

        async def reward_fn(prompts, completions, **kwargs):
            return rewards

        reference = CallableVerifier(
            lambda case: True,
            policy=ScorePolicy.zero_one(),
        )
        auditor = ShadowAuditor(reference)
        wrapped = wrap_trl_reward(reward_fn, auditor, sample_rate=0)

        result = await wrapped(["question"], ["answer"], ground_truth=["answer"])
        auditor.close()

        self.assertIs(result, rewards)


if __name__ == "__main__":
    unittest.main()
