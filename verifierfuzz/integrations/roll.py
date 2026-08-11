"""Alibaba ROLL reward-worker adapters without importing ROLL or torch."""

from __future__ import annotations

import functools
import hashlib
import inspect
import random
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from verifierfuzz.protocol import (
    ScorePolicy,
    VerifierCase,
    VerifierOutcome,
    outcome_from_raw,
)

from .shadow import ShadowAuditor


def _column(container: Any, key: str) -> Any:
    try:
        return container[key]
    except (KeyError, TypeError):
        return None


def _rows(values: Any, indices: Sequence[int]) -> List[Any]:
    return [values[index] for index in indices]


def _decode_rows(
    data: Any,
    *,
    tensor_key: str,
    text_key: str,
    indices: Sequence[int],
    tokenizer: Any,
    skip_special_tokens: bool,
) -> List[Any]:
    non_tensor_batch = getattr(data, "non_tensor_batch", {})
    text_values = _column(non_tensor_batch, text_key)
    if text_values is not None:
        return _rows(text_values, indices)

    batch = getattr(data, "batch", {})
    token_values = _column(batch, tensor_key)
    if token_values is None:
        return [""] * len(indices)
    selected = _rows(token_values, indices)
    if tokenizer is None:
        if all(isinstance(value, str) for value in selected):
            return selected
        raise ValueError(
            f"tokenizer is required to decode ROLL batch field {tensor_key!r}"
        )
    return list(
        tokenizer.batch_decode(
            selected,
            skip_special_tokens=skip_special_tokens,
        )
    )


def _item(values: Any, index: int, default: Any = None) -> Any:
    if values is None:
        return default
    try:
        return values[index]
    except (IndexError, KeyError, TypeError):
        return default


def _case_id(
    index: int,
    prompt: Any,
    response: Any,
    non_tensor_batch: Mapping[str, Any],
    id_key: str,
) -> str:
    value = _item(_column(non_tensor_batch, id_key), index)
    if value is not None:
        return str(value)
    payload = f"{prompt!r}\0{response!r}\0{index}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def make_roll_cases(
    data: Any,
    tokenizer: Any = None,
    *,
    indices: Optional[Sequence[int]] = None,
    prompt_tensor_key: str = "prompts",
    response_tensor_key: str = "responses",
    prompt_text_key: str = "prompt",
    response_text_key: str = "response",
    reference_key: str = "ground_truth",
    id_key: str = "id",
    skip_special_tokens: bool = False,
) -> List[VerifierCase]:
    """Convert selected rows from a ROLL ``DataProto`` into audit cases."""

    if indices is None:
        indices = list(range(len(data)))
    else:
        indices = list(indices)
    prompts = _decode_rows(
        data,
        tensor_key=prompt_tensor_key,
        text_key=prompt_text_key,
        indices=indices,
        tokenizer=tokenizer,
        skip_special_tokens=skip_special_tokens,
    )
    responses = _decode_rows(
        data,
        tensor_key=response_tensor_key,
        text_key=response_text_key,
        indices=indices,
        tokenizer=tokenizer,
        skip_special_tokens=skip_special_tokens,
    )
    non_tensor_batch = getattr(data, "non_tensor_batch", {})
    references = _column(non_tensor_batch, reference_key)
    meta_info = dict(getattr(data, "meta_info", {}) or {})

    cases = []
    for position, index in enumerate(indices):
        roll_fields: Dict[str, Any] = {}
        for key, values in non_tensor_batch.items():
            if key not in (prompt_text_key, response_text_key, reference_key):
                roll_fields[key] = _item(values, index)
        prompt = prompts[position]
        response = responses[position]
        cases.append(
            VerifierCase(
                case_id=_case_id(
                    index,
                    prompt,
                    response,
                    non_tensor_batch,
                    id_key,
                ),
                prompt=prompt,
                completion=response,
                reference=_item(references, index),
                metadata={
                    "framework": "roll",
                    "batch_index": index,
                    "roll_fields": roll_fields,
                    "meta_info": meta_info,
                },
            )
        )
    return cases


def _reward_values(output: Any, reward_key: str) -> Any:
    batch = getattr(output, "batch", {})
    values = _column(batch, reward_key)
    if values is None:
        raise KeyError(f"ROLL reward output does not contain {reward_key!r}")
    return values


class RollVerifier:
    """Audit a ROLL worker using a caller-provided one-case DataProto factory."""

    def __init__(
        self,
        worker_or_function: Any,
        data_factory: Callable[[VerifierCase], Any],
        *,
        policy: ScorePolicy = ScorePolicy(),
        reward_key: str = "response_level_rewards",
    ) -> None:
        self.function = getattr(
            worker_or_function,
            "compute_rewards",
            worker_or_function,
        )
        self.data_factory = data_factory
        self.policy = policy
        self.reward_key = reward_key

    async def evaluate(self, case: VerifierCase) -> VerifierOutcome:
        started = time.perf_counter()
        output = self.function(self.data_factory(case))
        if inspect.isawaitable(output):
            output = await output
        raw = _reward_values(output, self.reward_key)[0]
        return outcome_from_raw(
            raw,
            self.policy,
            duration_ms=(time.perf_counter() - started) * 1000,
        )


def wrap_roll_compute_rewards(
    function: Callable[..., Any],
    auditor: ShadowAuditor,
    tokenizer: Any = None,
    *,
    sample_rate: float = 0.0,
    random_seed: int = 0,
    reward_key: str = "response_level_rewards",
    prompt_tensor_key: str = "prompts",
    response_tensor_key: str = "responses",
    prompt_text_key: str = "prompt",
    response_text_key: str = "response",
    reference_key: str = "ground_truth",
    id_key: str = "id",
    skip_special_tokens: bool = False,
) -> Callable[..., Any]:
    """Wrap ``RewardWorker.compute_rewards`` and return its exact DataProto."""

    if not 0.0 <= sample_rate <= 1.0:
        raise ValueError("sample_rate must be between 0 and 1")
    rng = random.Random(random_seed)

    def submit(data: Any, output: Any) -> None:
        rewards = _reward_values(output, reward_key)
        selected = [
            index
            for index in range(len(rewards))
            if sample_rate and rng.random() < sample_rate
        ]
        if not selected:
            return
        cases = make_roll_cases(
            data,
            tokenizer,
            indices=selected,
            prompt_tensor_key=prompt_tensor_key,
            response_tensor_key=response_tensor_key,
            prompt_text_key=prompt_text_key,
            response_text_key=response_text_key,
            reference_key=reference_key,
            id_key=id_key,
            skip_special_tokens=skip_special_tokens,
        )
        for case, index in zip(cases, selected):
            auditor.try_submit(case, rewards[index])

    if inspect.iscoroutinefunction(function):

        @functools.wraps(function)
        async def async_wrapped(data: Any, *args: Any, **kwargs: Any) -> Any:
            output = await function(data, *args, **kwargs)
            try:
                submit(data, output)
            except Exception:
                auditor.record_error()
            return output

        return async_wrapped

    @functools.wraps(function)
    def wrapped(data: Any, *args: Any, **kwargs: Any) -> Any:
        output = function(data, *args, **kwargs)
        try:
            submit(data, output)
        except Exception:
            auditor.record_error()
        return output

    return wrapped
