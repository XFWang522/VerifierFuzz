"""slime --custom-rm-path example with fail-open shadow auditing."""

from verifierfuzz.integrations import (
    CallableVerifier,
    ShadowAuditor,
    wrap_slime_group_reward,
    wrap_slime_reward,
)
from verifierfuzz.protocol import ScorePolicy
from verifierfuzz.reporting import JsonlSink


async def target_reward(args, sample):
    return float(str(sample.label).strip() in sample.response)


strict_reference = CallableVerifier(
    lambda case: case.completion.strip() == str(case.reference).strip(),
    policy=ScorePolicy.zero_one(),
)
auditor = ShadowAuditor(
    strict_reference,
    target_policy=ScorePolicy.zero_one(),
    sink=JsonlSink("slime-shadow-findings.jsonl"),
)

# Use with: --custom-rm-path examples.slime_reward.custom_rm
custom_rm = wrap_slime_reward(target_reward, auditor, sample_rate=0.01)


async def target_group_reward(args, samples):
    return [await target_reward(args, sample) for sample in samples]


# Use together with --group-rm.
group_rm = wrap_slime_group_reward(
    target_group_reward,
    auditor,
    sample_rate=0.01,
)
