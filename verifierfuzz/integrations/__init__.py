"""RL framework and callable verifier adapters."""

from .callable import CallableVerifier
from .roll import RollVerifier, make_roll_cases, wrap_roll_compute_rewards
from .shadow import ShadowAuditor, ShadowStats
from .slime import (
    SlimeBatchVerifier,
    SlimeVerifier,
    make_slime_case,
    wrap_slime_group_reward,
    wrap_slime_reward,
)
from .trl import TrlVerifier, make_trl_case, wrap_trl_reward
from .verl import VerlVerifier, make_verl_case, wrap_verl_reward

__all__ = [
    "CallableVerifier",
    "RollVerifier",
    "ShadowAuditor",
    "ShadowStats",
    "SlimeBatchVerifier",
    "SlimeVerifier",
    "TrlVerifier",
    "VerlVerifier",
    "make_roll_cases",
    "make_slime_case",
    "make_trl_case",
    "make_verl_case",
    "wrap_slime_group_reward",
    "wrap_slime_reward",
    "wrap_roll_compute_rewards",
    "wrap_trl_reward",
    "wrap_verl_reward",
]
