"""verl custom reward function adapter and fail-open shadow wrapper."""

from __future__ import annotations

import functools
import hashlib
import inspect
import random
import time
from typing import Any, Callable, Dict, Mapping, Optional

from verifierfuzz.protocol import (
    ScorePolicy,
    VerifierCase,
    VerifierOutcome,
    outcome_from_raw,
)

from .shadow import ShadowAuditor


def _case_id(solution_str: str, extra_info: Optional[Mapping[str, Any]]) -> str:
    if extra_info:
        for key in ("data_uid", "uid", "id", "index"):
            if key in extra_info:
                return str(extra_info[key])
    return hashlib.sha256(solution_str.encode("utf-8")).hexdigest()[:16]


def make_verl_case(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[Mapping[str, Any]] = None,
) -> VerifierCase:
    info = dict(extra_info or {})
    return VerifierCase(
        case_id=_case_id(solution_str, info),
        prompt=info.get("question", info.get("prompt", "")),
        completion=solution_str,
        reference=ground_truth,
        metadata={
            "framework": "verl",
            "data_source": data_source,
            "extra_info": info,
        },
    )


class VerlVerifier:
    def __init__(
        self,
        function: Callable[..., Any],
        *,
        policy: ScorePolicy = ScorePolicy(),
        reward_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.function = function
        self.policy = policy
        self.reward_kwargs: Dict[str, Any] = dict(reward_kwargs or {})

    async def evaluate(self, case: VerifierCase) -> VerifierOutcome:
        started = time.perf_counter()
        metadata = case.metadata
        raw = self.function(
            data_source=metadata.get("data_source", ""),
            solution_str=case.completion,
            ground_truth=case.reference,
            extra_info=metadata.get("extra_info", {}),
            **self.reward_kwargs,
        )
        if inspect.isawaitable(raw):
            raw = await raw
        return outcome_from_raw(
            raw,
            self.policy,
            duration_ms=(time.perf_counter() - started) * 1000,
        )


def wrap_verl_reward(
    function: Callable[..., Any],
    auditor: ShadowAuditor,
    *,
    sample_rate: float = 0.0,
    random_seed: int = 0,
) -> Callable[..., Any]:
    """Return a verl-compatible reward function with sampled shadow auditing."""

    if not 0.0 <= sample_rate <= 1.0:
        raise ValueError("sample_rate must be between 0 and 1")
    rng = random.Random(random_seed)

    def submit(
        raw: Any,
        data_source: str,
        solution_str: str,
        ground_truth: Any,
        extra_info: Optional[Mapping[str, Any]],
    ) -> None:
        if sample_rate and rng.random() < sample_rate:
            try:
                auditor.try_submit(
                    make_verl_case(
                        data_source,
                        solution_str,
                        ground_truth,
                        extra_info,
                    ),
                    raw,
                )
            except Exception:
                auditor.record_error()

    if inspect.iscoroutinefunction(function):
        @functools.wraps(function)
        async def async_wrapped(
            data_source: str,
            solution_str: str,
            ground_truth: Any,
            extra_info: Optional[Mapping[str, Any]] = None,
            **kwargs: Any,
        ) -> Any:
            raw = await function(
                data_source=data_source,
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **kwargs,
            )
            submit(raw, data_source, solution_str, ground_truth, extra_info)
            return raw

        return async_wrapped

    @functools.wraps(function)
    def wrapped(
        data_source: str,
        solution_str: str,
        ground_truth: Any,
        extra_info: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        raw = function(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            **kwargs,
        )
        submit(raw, data_source, solution_str, ground_truth, extra_info)
        return raw

    return wrapped
