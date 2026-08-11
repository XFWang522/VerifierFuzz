"""ROLL RewardWorker subclass with sampled shadow auditing."""

from roll.distributed.scheduler.decorator import Dispatch, register
from roll.pipeline.rlvr.rewards.math_rule_reward_worker import MathRuleRewardWorker

from verifierfuzz.integrations import (
    CallableVerifier,
    ShadowAuditor,
    wrap_roll_compute_rewards,
)
from verifierfuzz.protocol import ScorePolicy
from verifierfuzz.reporting import JsonlSink


class AuditedMathRewardWorker(MathRuleRewardWorker):
    def __init__(self, worker_config):
        super().__init__(worker_config)
        reference = CallableVerifier(
            lambda case: case.completion.strip()
            == str(case.reference).strip(),
            policy=ScorePolicy.zero_one(),
        )
        auditor = ShadowAuditor(
            reference,
            target_policy=ScorePolicy.zero_one(),
            sink=JsonlSink(
                f"roll-shadow-findings-rank-{self.rank_info.rank}.jsonl"
            ),
        )
        self._audited_compute_rewards = wrap_roll_compute_rewards(
            super().compute_rewards,
            auditor,
            self.tokenizer,
            sample_rate=0.01,
        )

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False)
    def compute_rewards(self, data):
        return self._audited_compute_rewards(data)
