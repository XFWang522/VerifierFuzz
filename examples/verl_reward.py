"""Minimal verl reward function with VerifierFuzz shadow auditing."""

from verifierfuzz.integrations import (
    ShadowAuditor,
    VerlVerifier,
    wrap_verl_reward,
)
from verifierfuzz.protocol import ScorePolicy
from verifierfuzz.reporting import JsonlSink


def target_reward(data_source, solution_str, ground_truth, extra_info=None):
    return 1 if str(ground_truth) in solution_str else -1


def strict_reference(data_source, solution_str, ground_truth, extra_info=None):
    return 1 if solution_str.strip() == str(ground_truth).strip() else -1


_auditor = ShadowAuditor(
    VerlVerifier(strict_reference, policy=ScorePolicy.signed()),
    target_policy=ScorePolicy.signed(),
    sink=JsonlSink("verifierfuzz-shadow.jsonl"),
)

# Configure verl custom_reward_function.path to this file and
# custom_reward_function.name to compute_score.
compute_score = wrap_verl_reward(
    target_reward,
    _auditor,
    sample_rate=0.01,
)
