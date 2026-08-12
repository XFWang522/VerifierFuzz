"""Stream real reward datasets into framework-independent verifier cases."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Mapping, Optional, Sequence

from verifierfuzz.protocol import VerifierCase


_MISSING = object()


@dataclass(frozen=True)
class DatasetColumns:
    prompt: str = "prompt"
    completion: str = "response"
    reference: str = "ground_truth"
    case_id: Optional[str] = "id"
    metadata: Sequence[str] = ()


def _value_at(row: Mapping[str, Any], path: Optional[str], default: Any = _MISSING) -> Any:
    if path is None:
        return default
    value: Any = row
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            if default is _MISSING:
                raise KeyError(path)
            return default
        value = value[part]
    return value


def _iter_jsonl(path: Path) -> Iterator[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {error}"
                ) from error
            if not isinstance(row, Mapping):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            yield row


def _iter_json(path: Path) -> Iterator[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"{path}: JSON dataset must contain an array")
    for index, row in enumerate(payload):
        if not isinstance(row, Mapping):
            raise ValueError(f"{path}: row {index} must be an object")
        yield row


def _iter_parquet(path: Path) -> Iterator[Mapping[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(
            "Parquet input requires pyarrow; install it with 'pip install pyarrow'"
        ) from error
    parquet_file = parquet.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=1024):
        yield from batch.to_pylist()


def iter_dataset_rows(
    path: str,
    *,
    format: str = "auto",
) -> Iterable[Mapping[str, Any]]:
    dataset_path = Path(path)
    selected_format = format
    if selected_format == "auto":
        suffix = dataset_path.suffix.lower()
        selected_format = {
            ".json": "json",
            ".jsonl": "jsonl",
            ".parquet": "parquet",
        }.get(suffix, "jsonl")
    if selected_format == "jsonl":
        return _iter_jsonl(dataset_path)
    if selected_format == "json":
        return _iter_json(dataset_path)
    if selected_format == "parquet":
        return _iter_parquet(dataset_path)
    raise ValueError(f"unsupported dataset format: {selected_format}")


def load_dataset_cases(
    path: str,
    *,
    columns: DatasetColumns = DatasetColumns(),
    format: str = "auto",
    framework: str = "generic",
    offset: int = 0,
    limit: Optional[int] = None,
) -> List[VerifierCase]:
    if offset < 0:
        raise ValueError("offset must be non-negative")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")

    cases = []
    for index, row in enumerate(iter_dataset_rows(path, format=format)):
        if index < offset:
            continue
        if limit is not None and len(cases) >= limit:
            break
        metadata = {
            "framework": framework,
            "dataset_path": str(Path(path)),
            "dataset_index": index,
        }
        for metadata_path in columns.metadata:
            value = _value_at(row, metadata_path, _MISSING)
            if value is not _MISSING:
                metadata[metadata_path] = value
        case_id = _value_at(row, columns.case_id, None)
        cases.append(
            VerifierCase(
                case_id=str(case_id) if case_id is not None else f"row-{index}",
                prompt=_value_at(row, columns.prompt, ""),
                completion=_value_at(row, columns.completion),
                reference=_value_at(row, columns.reference, None),
                metadata=metadata,
            )
        )
    return cases
