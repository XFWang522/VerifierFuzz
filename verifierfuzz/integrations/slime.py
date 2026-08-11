"""THUDM slime custom reward adapters without importing slime or torch."""

from __future__ import annotations

import hashlib
import inspect
import random
import time
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from verifierfuzz.protocol import (
    ScorePolicy,
    VerifierCase,
    VerifierOutcome,
    outcome_from_raw,
)

from .shadow import ShadowAuditor


def _sample_case_id(sample: Any) -> str:
    for name in ("rollout_id", "index", "group_index", "session_id"):
        value = getattr(sample, name, None)
        if value is not None:
            return str(value)
    payload = f"{getattr(sample, 'prompt', '')!r}\0{getattr(sample, 'response', '')!r}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def make_slime_case(args: Any, sample: Any) -> VerifierCase:
    """Convert a duck-typed slime ``Sample`` into the public case protocol."""

    sample_metadata = getattr(sample, "metadata", {})
    if not isinstance(sample_metadata, Mapping):
        sample_metadata = {}
    return VerifierCase(
        case_id=_sample_case_id(sample),
        prompt=getattr(sample, "prompt", ""),
        completion=getattr(sample, "response", ""),
        reference=getattr(sample, "label", None),
        metadata={
            "framework": "slime",
            "sample_metadata": dict(sample_metadata),
            "group_index": getattr(sample, "group_index", None),
            "index": getattr(sample, "index", None),
            "rollout_id": getattr(sample, "rollout_id", None),
            "response_length": getattr(sample, "response_length", None),
            "reward_key": getattr(args, "reward_key", None),
        },
    )


def _sample_from_case(case: VerifierCase) -> Any:
    metadata = case.metadata
    return SimpleNamespace(
        prompt=case.prompt,
        response=case.completion,
        label=case.reference,
        metadata=dict(metadata.get("sample_metadata", {})),
        group_index=metadata.get("group_index"),
        index=metadata.get("index", case.case_id),
        rollout_id=metadata.get("rollout_id"),
        response_length=metadata.get("response_length", 0),
        reward=None,
        custom_rm_path=None,
    )


def _select_reward(raw: Any, reward_key: Optional[str]) -> Any:
    if isinstance(raw, Mapping) and reward_key:
        if reward_key not in raw:
            raise KeyError(f"slime reward mapping does not contain {reward_key!r}")
        return raw[reward_key]
    return raw


class SlimeVerifier:
    """Adapt ``async def custom_rm(args, sample, **kwargs)``."""

    def __init__(
        self,
        function: Callable[..., Any],
        args: Any = None,
        *,
        policy: ScorePolicy = ScorePolicy(),
        reward_key: Optional[str] = None,
        reward_kwargs: Optional[Mapping[str, Any]] = None,
        sample_factory: Optional[Callable[[VerifierCase], Any]] = None,
    ) -> None:
        self.function = function
        self.args = args if args is not None else SimpleNamespace()
        self.policy = policy
        self.reward_key = reward_key or getattr(self.args, "reward_key", None)
        self.reward_kwargs: Dict[str, Any] = dict(reward_kwargs or {})
        self.sample_factory = sample_factory or _sample_from_case

    async def evaluate(self, case: VerifierCase) -> VerifierOutcome:
        started = time.perf_counter()
        raw = self.function(
            self.args,
            self.sample_factory(case),
            **self.reward_kwargs,
        )
        if inspect.isawaitable(raw):
            raw = await raw
        selected = _select_reward(raw, self.reward_key)
        outcome = outcome_from_raw(
            selected,
            self.policy,
            duration_ms=(time.perf_counter() - started) * 1000,
        )
        return replace(outcome, raw=raw)


class SlimeBatchVerifier(SlimeVerifier):
    """Adapt slime ``--group-rm`` batch custom reward functions."""

    async def evaluate(self, case: VerifierCase) -> VerifierOutcome:
        started = time.perf_counter()
        raw_batch = self.function(
            self.args,
            [self.sample_factory(case)],
            **self.reward_kwargs,
        )
        if inspect.isawaitable(raw_batch):
            raw_batch = await raw_batch
        raw = raw_batch[0]
        selected = _select_reward(raw, self.reward_key)
        outcome = outcome_from_raw(
            selected,
            self.policy,
            duration_ms=(time.perf_counter() - started) * 1000,
        )
        return replace(outcome, raw=raw)


def wrap_slime_reward(
    function: Callable[..., Any],
    auditor: ShadowAuditor,
    *,
    sample_rate: float = 0.0,
    random_seed: int = 0,
    reward_key: Optional[str] = None,
) -> Callable[..., Any]:
    """Wrap slime single-sample ``--custom-rm-path`` functions."""

    if not 0.0 <= sample_rate <= 1.0:
        raise ValueError("sample_rate must be between 0 and 1")
    rng = random.Random(random_seed)

    async def wrapped(args: Any, sample: Any, **kwargs: Any) -> Any:
        raw = function(args, sample, **kwargs)
        if inspect.isawaitable(raw):
            raw = await raw
        if sample_rate and rng.random() < sample_rate:
            key = reward_key or getattr(args, "reward_key", None)
            auditor.try_submit(
                make_slime_case(args, sample),
                _select_reward(raw, key),
            )
        return raw

    return wrapped


def wrap_slime_group_reward(
    function: Callable[..., Any],
    auditor: ShadowAuditor,
    *,
    sample_rate: float = 0.0,
    random_seed: int = 0,
    reward_key: Optional[str] = None,
) -> Callable[..., Any]:
    """Wrap slime ``--group-rm`` functions while preserving reward order."""

    if not 0.0 <= sample_rate <= 1.0:
        raise ValueError("sample_rate must be between 0 and 1")
    rng = random.Random(random_seed)

    async def wrapped(
        args: Any,
        samples: Sequence[Any],
        **kwargs: Any,
    ) -> Any:
        raw_batch = function(args, samples, **kwargs)
        if inspect.isawaitable(raw_batch):
            raw_batch = await raw_batch
        if len(raw_batch) != len(samples):
            raise ValueError("slime group reward count must match sample count")
        key = reward_key or getattr(args, "reward_key", None)
        for sample, raw in zip(samples, raw_batch):
            if sample_rate and rng.random() < sample_rate:
                auditor.try_submit(
                    make_slime_case(args, sample),
                    _select_reward(raw, key),
                )
        return raw_batch

    return wrapped
