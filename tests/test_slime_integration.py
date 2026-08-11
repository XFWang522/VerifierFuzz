import unittest
from types import SimpleNamespace

from verifierfuzz.integrations import (
    CallableVerifier,
    ShadowAuditor,
    SlimeBatchVerifier,
    SlimeVerifier,
    make_slime_case,
    wrap_slime_group_reward,
    wrap_slime_reward,
)
from verifierfuzz.protocol import Decision, ScorePolicy, VerifierCase


class SlimeIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_sample_adapter_uses_official_signature(self):
        calls = []
        args = SimpleNamespace(reward_key=None)

        async def reward_fn(call_args, sample, scale):
            calls.append((call_args, sample, scale))
            return 0.5 * scale

        verifier = SlimeVerifier(
            reward_fn,
            args,
            policy=ScorePolicy(pass_threshold=0.75),
            reward_kwargs={"scale": 2},
        )
        outcome = await verifier.evaluate(
            VerifierCase(
                case_id="slime-1",
                prompt="question",
                completion="answer",
                reference="answer",
                metadata={"sample_metadata": {"source": "math"}},
            )
        )

        self.assertEqual(outcome.decision, Decision.PASS)
        self.assertIs(calls[0][0], args)
        self.assertEqual(calls[0][1].response, "answer")
        self.assertEqual(calls[0][2], 2)

    async def test_batch_adapter_selects_configured_reward_key(self):
        args = SimpleNamespace(reward_key="accuracy")

        async def reward_fn(call_args, samples):
            return [{"accuracy": 1.0, "format": 0.0}]

        verifier = SlimeBatchVerifier(
            reward_fn,
            args,
            policy=ScorePolicy.zero_one(),
        )
        outcome = await verifier.evaluate(
            VerifierCase(case_id="slime-group", completion="answer")
        )

        self.assertEqual(outcome.decision, Decision.PASS)
        self.assertEqual(outcome.raw["format"], 0.0)

    async def test_single_wrapper_returns_exact_original_reward(self):
        raw_reward = {"accuracy": 1.0, "format": 0.0}

        async def reward_fn(args, sample):
            return raw_reward

        findings = []
        auditor = ShadowAuditor(
            CallableVerifier(
                lambda case: False,
                policy=ScorePolicy.zero_one(),
            ),
            target_policy=ScorePolicy.zero_one(),
            sink=findings.append,
        )
        wrapped = wrap_slime_reward(
            reward_fn,
            auditor,
            sample_rate=1.0,
            reward_key="accuracy",
        )
        sample = SimpleNamespace(
            index=7,
            prompt="question",
            response="wrong answer",
            label="right answer",
            metadata={"source": "math"},
        )

        result = await wrapped(SimpleNamespace(reward_key="accuracy"), sample)
        auditor.close()

        self.assertIs(result, raw_reward)
        self.assertEqual(findings[0].kind, "false_positive")
        self.assertEqual(findings[0].case.case_id, "7")

    async def test_group_wrapper_preserves_batch_and_order(self):
        raw_rewards = [1.0, 0.0]

        async def reward_fn(args, samples):
            return raw_rewards

        auditor = ShadowAuditor(
            CallableVerifier(
                lambda case: case.completion == case.reference,
                policy=ScorePolicy.zero_one(),
            ),
            target_policy=ScorePolicy.zero_one(),
        )
        wrapped = wrap_slime_group_reward(
            reward_fn,
            auditor,
            sample_rate=1.0,
        )
        samples = [
            SimpleNamespace(
                index=1,
                prompt="q1",
                response="a1",
                label="a1",
                metadata={},
            ),
            SimpleNamespace(
                index=2,
                prompt="q2",
                response="bad",
                label="a2",
                metadata={},
            ),
        ]

        result = await wrapped(SimpleNamespace(reward_key=None), samples)
        auditor.close()

        self.assertIs(result, raw_rewards)
        self.assertEqual(auditor.stats().processed, 2)

    async def test_group_wrapper_rejects_misaligned_rewards(self):
        async def reward_fn(args, samples):
            return []

        auditor = ShadowAuditor(CallableVerifier(lambda case: True))
        wrapped = wrap_slime_group_reward(reward_fn, auditor)

        with self.assertRaisesRegex(ValueError, "count must match"):
            await wrapped(
                SimpleNamespace(reward_key=None),
                [SimpleNamespace(prompt="", response="", label=None, metadata={})],
            )
        auditor.close()

    async def test_audit_conversion_failure_is_fail_open(self):
        raw_reward = {"accuracy": 1.0}

        async def reward_fn(args, sample):
            return raw_reward

        auditor = ShadowAuditor(CallableVerifier(lambda case: True))
        wrapped = wrap_slime_reward(
            reward_fn,
            auditor,
            sample_rate=1.0,
            reward_key="missing",
        )

        result = await wrapped(
            SimpleNamespace(reward_key=None),
            SimpleNamespace(prompt="", response="", label=None, metadata={}),
        )
        auditor.close()

        self.assertIs(result, raw_reward)
        self.assertEqual(auditor.stats().errors, 1)

    def test_make_case_uses_slime_sample_fields(self):
        case = make_slime_case(
            SimpleNamespace(reward_key=None),
            SimpleNamespace(
                rollout_id=11,
                index=3,
                group_index=2,
                prompt=[{"role": "user", "content": "question"}],
                response="answer",
                label={"ground_truth": "answer"},
                response_length=5,
                metadata={"rm_type": "math"},
            ),
        )

        self.assertEqual(case.case_id, "11")
        self.assertEqual(case.completion, "answer")
        self.assertEqual(case.metadata["sample_metadata"]["rm_type"], "math")


if __name__ == "__main__":
    unittest.main()
