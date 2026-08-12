"""Read-only probes for upstream RL framework reward contracts."""

from __future__ import annotations

import ast
import urllib.request
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Sequence, Union


SourceFetcher = Callable[[str], str]
FunctionNode = Union[ast.FunctionDef, ast.AsyncFunctionDef]


@dataclass(frozen=True)
class UpstreamContract:
    framework: str
    url: str


@dataclass(frozen=True)
class ContractResult:
    framework: str
    url: str
    passed: bool
    checks: Sequence[str] = ()
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


UPSTREAM_CONTRACTS = (
    UpstreamContract(
        framework="verl",
        url=(
            "https://raw.githubusercontent.com/volcengine/verl/main/"
            "verl/trainer/ppo/reward.py"
        ),
    ),
    UpstreamContract(
        framework="slime",
        url=(
            "https://raw.githubusercontent.com/THUDM/slime/main/"
            "slime/rollout/rm_hub/__init__.py"
        ),
    ),
    UpstreamContract(
        framework="roll",
        url=(
            "https://raw.githubusercontent.com/alibaba/ROLL/main/"
            "roll/pipeline/rlvr/rewards/math_rule_reward_worker.py"
        ),
    ),
)


def fetch_source(url: str) -> str:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "VerifierFuzz-contract-probe"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def _find_function(
    tree: ast.Module,
    name: str,
    *,
    class_name: Optional[str] = None,
) -> FunctionNode:
    body = tree.body
    if class_name is not None:
        matching_classes = [
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ]
        if not matching_classes:
            raise ValueError(f"missing class {class_name}")
        body = matching_classes[0].body
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == name:
                return node
    location = f"{class_name}.{name}" if class_name else name
    raise ValueError(f"missing function {location}")


def _argument_names(function: FunctionNode) -> List[str]:
    return [
        argument.arg
        for argument in (
            list(function.args.posonlyargs) + list(function.args.args)
        )
    ]


def _require_arguments(
    function: FunctionNode,
    expected: Sequence[str],
    *,
    require_async: bool,
    require_kwargs: bool = False,
) -> str:
    actual = _argument_names(function)
    if actual[: len(expected)] != list(expected):
        raise ValueError(
            f"{function.name} positional arguments changed: "
            f"expected {list(expected)}, got {actual}"
        )
    if require_async and not isinstance(function, ast.AsyncFunctionDef):
        raise ValueError(f"{function.name} is no longer async")
    if require_kwargs and (
        function.args.kwarg is None or function.args.kwarg.arg != "kwargs"
    ):
        raise ValueError(f"{function.name} no longer accepts **kwargs")
    return f"{function.name}({', '.join(expected)})"


def check_contract_source(framework: str, source: str) -> Sequence[str]:
    tree = ast.parse(source)
    if framework == "verl":
        function = _find_function(tree, "get_custom_reward_fn")
        checks = [
            _require_arguments(
                function,
                ["config"],
                require_async=False,
            )
        ]
        for token in (
            "custom_reward_function",
            "reward_kwargs",
            "inspect.iscoroutinefunction",
            "_call_with_kwargs_async",
        ):
            if token not in source:
                raise ValueError(f"get_custom_reward_fn no longer contains {token!r}")
            checks.append(token)
        return checks
    if framework == "slime":
        single = _find_function(tree, "async_rm")
        batch = _find_function(tree, "batched_async_rm")
        return [
            _require_arguments(
                single,
                ["args", "sample"],
                require_async=True,
                require_kwargs=True,
            ),
            _require_arguments(
                batch,
                ["args", "samples"],
                require_async=True,
                require_kwargs=True,
            ),
        ]
    if framework == "roll":
        function = _find_function(
            tree,
            "compute_rewards",
            class_name="MathRuleRewardWorker",
        )
        return [
            _require_arguments(
                function,
                ["self", "data"],
                require_async=False,
            )
        ]
    raise ValueError(f"unknown framework contract: {framework}")


def check_upstream_contracts(
    *,
    fetcher: SourceFetcher = fetch_source,
    contracts: Sequence[UpstreamContract] = UPSTREAM_CONTRACTS,
) -> List[ContractResult]:
    results = []
    for contract in contracts:
        try:
            checks = check_contract_source(
                contract.framework,
                fetcher(contract.url),
            )
            results.append(
                ContractResult(
                    framework=contract.framework,
                    url=contract.url,
                    passed=True,
                    checks=checks,
                )
            )
        except Exception as error:
            results.append(
                ContractResult(
                    framework=contract.framework,
                    url=contract.url,
                    passed=False,
                    error=f"{type(error).__name__}: {error}",
                )
            )
    return results
