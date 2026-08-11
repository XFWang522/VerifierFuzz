"""Bounded, fail-open shadow auditing for live training loops."""

from __future__ import annotations

import asyncio
import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

from verifierfuzz.engine import AuditFinding, classify_disagreement, evaluate_verifier
from verifierfuzz.protocol import (
    ScorePolicy,
    VerifierCase,
    error_outcome,
    outcome_from_raw,
)


@dataclass(frozen=True)
class ShadowStats:
    submitted: int
    dropped: int
    processed: int
    errors: int


class ShadowAuditor:
    """Evaluate a reference off-path without changing or delaying rewards."""

    _STOP = object()

    def __init__(
        self,
        reference: Any,
        *,
        target_policy: ScorePolicy = ScorePolicy(),
        sink: Optional[Callable[[AuditFinding], None]] = None,
        max_queue_size: int = 128,
        reference_timeout: Optional[float] = None,
    ) -> None:
        if max_queue_size < 1:
            raise ValueError("max_queue_size must be at least 1")
        if reference_timeout is not None and reference_timeout <= 0:
            raise ValueError("reference_timeout must be positive")
        self.reference = reference
        self.target_policy = target_policy
        self.sink = sink or (lambda finding: None)
        self.reference_timeout = reference_timeout
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=max_queue_size)
        self._lock = threading.Lock()
        self._submitted = 0
        self._dropped = 0
        self._processed = 0
        self._errors = 0
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="verifierfuzz-shadow",
            daemon=True,
        )
        self._thread.start()

    def try_submit(self, case: VerifierCase, raw_target: Any) -> bool:
        if self._closed:
            return False
        try:
            self._queue.put_nowait((case, raw_target))
        except queue.Full:
            with self._lock:
                self._dropped += 1
            return False
        with self._lock:
            self._submitted += 1
        return True

    def stats(self) -> ShadowStats:
        with self._lock:
            return ShadowStats(
                submitted=self._submitted,
                dropped=self._dropped,
                processed=self._processed,
                errors=self._errors,
            )

    def record_error(self) -> None:
        """Record an off-path adapter failure without raising into training."""

        with self._lock:
            self._errors += 1

    def close(self, *, drain: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        if drain:
            self._queue.join()
        else:
            while True:
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                except queue.Empty:
                    break
        self._queue.put(self._STOP)
        self._thread.join(timeout=5)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._STOP:
                    return
                case, raw_target = item
                self._process(case, raw_target)
            finally:
                self._queue.task_done()

    def _process(self, case: VerifierCase, raw_target: Any) -> None:
        try:
            target_outcome = outcome_from_raw(raw_target, self.target_policy)
            reference_outcome = asyncio.run(self._evaluate_reference(case))
            kind = classify_disagreement(target_outcome, reference_outcome)
            if kind is not None:
                self.sink(
                    AuditFinding(
                        case=case,
                        target_outcome=target_outcome,
                        reference_outcome=reference_outcome,
                        kind=kind,
                        relation="shadow",
                    )
                )
            with self._lock:
                self._processed += 1
        except Exception:
            with self._lock:
                self._errors += 1

    async def _evaluate_reference(self, case: VerifierCase):
        evaluation = evaluate_verifier(self.reference, case)
        if self.reference_timeout is None:
            return await evaluation
        try:
            return await asyncio.wait_for(
                evaluation,
                timeout=self.reference_timeout,
            )
        except asyncio.TimeoutError as error:
            return error_outcome(error)
