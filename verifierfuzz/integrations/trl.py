"""TRL GRPO custom reward adapter and fail-open shadow wrapper."""

from __future__ import annotations

import functools
import hashlib
import inspect
import random
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from verifierfuzz.protocol import (
    Decision,
    ScorePolicy,
    VerifierCase,
    VerifierOutcome,
    outcome_from_raw,
)

from .shadow import ShadowAuditor


def _stable_case_id(prompt: Any, completion: Any, index: int) -> str:
    payload = f"{prompt!r}\0{completion!r}\0{index}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _item_kwargs(
    kwargs: Mapping[str, Any],
    index: int,
    batch_size: int,
) -> Dict[str, Any]:
    item = {}
    for key, value in kwargs.items():
        if (
            not isinstance(value, (str, bytes, Mapping))
            and hasattr(value, "__len__")
            and hasattr(value, "__getitem__")
        ):
            try:
                if len(value) == batch_size:
                    item[key] = value[index]
                    continue
            except TypeError:
                pass
        item[key] = value
    return item


def make_trl_case(
    prompt: Any,
    completion: Any,
    *,
    index: int = 0,
    dataset_columns: Optional[Mapping[str, Any]] = None,
) -> VerifierCase:
    columns = dict(dataset_columns or {})
    reference = columns.get(
        "ground_truth",
        columns.get("reference", columns.get("answer")),
    )
    return VerifierCase(
        case_id=str(columns.get("case_id") or _stable_case_id(prompt, completion, index)),
        prompt=prompt,
        completion=completion,
        reference=reference,
        metadata={
            "framework": "trl",
            "dataset_columns": columns,
        },
    )


class TrlVerifier:
    def __init__(
        self,
        function: Callable[..., Any],
        *,
        policy: ScorePolicy = ScorePolicy(),
    ) -> None:
        self.function = function
        self.policy = policy

    async def evaluate(self, case: VerifierCase) -> VerifierOutcome:
        started = time.perf_counter()
        columns = dict(case.metadata.get("dataset_columns", {}))
        kwargs = {
            key: value if key == "trainer_state" else [value]
            for key, value in columns.items()
        }
        raw_batch = self.function(
            prompts=[case.prompt],
            completions=[case.completion],
            **kwargs,
        )
        if inspect.isawaitable(raw_batch):
            raw_batch = await raw_batch
        raw = raw_batch[0]
        if raw is None:
            return VerifierOutcome(
                decision=Decision.ABSTAIN,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        return outcome_from_raw(
            raw,
            self.policy,
            duration_ms=(time.perf_counter() - started) * 1000,
        )


def wrap_trl_reward(
    function: Callable[..., Any],
    auditor: ShadowAuditor,
    *,
    sample_rate: float = 0.0,
    random_seed: int = 0,
) -> Callable[..., Any]:
    """Return a TRL-compatible batch reward function with shadow auditing."""

    if not 0.0 <= sample_rate <= 1.0:
        raise ValueError("sample_rate must be between 0 and 1")
    rng = random.Random(random_seed)

    def submit_batch(
        raw_batch: Sequence[Any],
        prompts: Sequence[Any],
        completions: Sequence[Any],
        kwargs: Mapping[str, Any],
    ) -> None:
        batch_size = len(completions)
        for index, raw in enumerate(raw_batch):
            if raw is None or not sample_rate or rng.random() >= sample_rate:
                continue
            columns = _item_kwargs(kwargs, index, batch_size)
            auditor.try_submit(
                make_trl_case(
                    prompts[index],
                    completions[index],
                    index=index,
                    dataset_columns=columns,
                ),
                raw,
            )

    if inspect.iscoroutinefunction(function):
        @functools.wraps(function)
        async def async_wrapped(
            prompts: Sequence[Any],
            completions: Sequence[Any],
            **kwargs: Any,
        ) -> Any:
            raw_batch = await function(
                prompts=prompts,
                completions=completions,
                **kwargs,
            )
            submit_batch(raw_batch, prompts, completions, kwargs)
            return raw_batch

        return async_wrapped

    @functools.wraps(function)
    def wrapped(
        prompts: Sequence[Any],
        completions: Sequence[Any],
        **kwargs: Any,
    ) -> Any:
        raw_batch = function(
            prompts=prompts,
            completions=completions,
            **kwargs,
        )
        submit_batch(raw_batch, prompts, completions, kwargs)
        return raw_batch

    return wrapped
