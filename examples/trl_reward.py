"""Minimal TRL GRPO reward function with VerifierFuzz shadow auditing."""

from verifierfuzz.integrations import (
    ShadowAuditor,
    TrlVerifier,
    wrap_trl_reward,
)
from verifierfuzz.protocol import ScorePolicy
from verifierfuzz.reporting import JsonlSink


def target_reward(prompts, completions, ground_truth, **kwargs):
    return [
        1.0 if str(answer) in str(completion) else 0.0
        for completion, answer in zip(completions, ground_truth)
    ]


def strict_reference(prompts, completions, ground_truth, **kwargs):
    return [
        1.0 if str(completion).strip() == str(answer).strip() else 0.0
        for completion, answer in zip(completions, ground_truth)
    ]


_auditor = ShadowAuditor(
    TrlVerifier(strict_reference, policy=ScorePolicy.zero_one()),
    target_policy=ScorePolicy.zero_one(),
    sink=JsonlSink("verifierfuzz-shadow.jsonl"),
)

reward_func = wrap_trl_reward(
    target_reward,
    _auditor,
    sample_rate=0.01,
)
