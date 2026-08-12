# VerifierFuzz

**Find reward exploits before your model learns them.**

VerifierFuzz is a framework-independent audit SDK for RL reward functions. It
mutates completions, compares the training verifier with an independent
reference, minimizes disagreements, and freezes confirmed cases into a
regression corpus.

It does not replace the reward function. Pre-training audits gate risky
verifiers, live wrappers observe a sample of training traffic without changing
the returned reward, and regression tests prevent fixed exploits from returning.

## Why

In reinforcement learning with verifiable rewards, the verifier is part of the
training objective. A false positive is not merely an evaluation bug: repeated
optimization can teach the policy to exploit it.

VerifierFuzz provides:

- explicit score policies for `0/1`, `-1/1`, and continuous rewards;
- sync, async, and batch verifier contracts;
- deterministic mutation lineage and text boundary mutations;
- differential checking against independent reference oracles;
- minimization of reward-positive incorrect completions;
- JSONL and SARIF artifacts;
- native verl, slime, ROLL, and TRL reward adapters;
- bounded, fail-open shadow auditing for training loops.

## Quick start

The core package has no runtime dependencies:

```bash
python3 -m verifierfuzz demo
```

Run an audit against functions in a local Python file:

```bash
python3 -m verifierfuzz audit \
  --adapter pair \
  --oracle-adapter pair \
  --target-policy zero-one \
  --oracle-policy zero-one \
  --target examples/simple_rewards.py:target \
  --oracle examples/simple_rewards.py:oracle \
  --corpus examples/cases.jsonl \
  --minimize \
  --output findings.jsonl \
  --sarif findings.sarif \
  --max-findings 10
```

Replay frozen expectations in CI:

```bash
python3 -m verifierfuzz regression \
  --adapter pair \
  --oracle-adapter pair \
  --target-policy zero-one \
  --oracle-policy zero-one \
  --target examples/simple_rewards.py:target \
  --oracle examples/simple_rewards.py:oracle \
  --corpus examples/regression.jsonl
```

`audit` and `replay` return exit code `1` when findings exceed
`--max-findings`. `regression` returns `3` when an expected disagreement changes.

## RL framework integration

### verl

verl custom rewards use:

```python
compute_score(data_source, solution_str, ground_truth, extra_info=None)
```

`verifierfuzz.integrations.wrap_verl_reward` returns a function with that
contract. It preserves the original reward object and submits sampled cases to
a bounded shadow queue after the target reward returns. See
[`examples/verl_reward.py`](examples/verl_reward.py).

Both synchronous rewards and the asynchronous Reward Loop contract are
supported. Fixed `reward_kwargs`, such as a reward router address or tokenizer,
can be supplied to `VerlVerifier` for offline audits.

### slime

slime custom rewards use one of these asynchronous contracts:

```python
custom_rm(args, sample) -> float | dict
batched_custom_rm(args, samples) -> list[float | dict]  # --group-rm
```

Use `wrap_slime_reward` with `--custom-rm-path`, or
`wrap_slime_group_reward` together with `--group-rm`. Both wrappers return the
original reward object and support slime's keyed reward dictionaries through
`reward_key`. See [`examples/slime_reward.py`](examples/slime_reward.py).

```bash
--custom-rm-path examples.slime_reward.custom_rm
# or:
--group-rm --custom-rm-path examples.slime_reward.group_rm
```

### ROLL

ROLL RLVR rewards are worker methods:

```python
RewardWorker.compute_rewards(data: DataProto) -> DataProto
```

`wrap_roll_compute_rewards` samples rows from `prompts`, `responses`, and
`non_tensor_batch`, then submits normalized cases after the worker has produced
`response_level_rewards`. It returns the exact original `DataProto` object and
does not import ROLL or torch. See
[`examples/roll_reward_worker.py`](examples/roll_reward_worker.py) for a
subclass that can be referenced from `rewards.*.worker_cls`.

```yaml
rewards:
  math_rule:
    worker_cls: examples.roll_reward_worker.AuditedMathRewardWorker
```

### TRL

TRL GRPO rewards use:

```python
reward_func(prompts, completions, **dataset_columns) -> list[float | None]
```

`verifierfuzz.integrations.wrap_trl_reward` preserves the original batch object
and independently audits selected completions. Standard and conversational
prompt/completion values are passed through unchanged. See
[`examples/trl_reward.py`](examples/trl_reward.py).

### Compatibility and failure isolation

The adapters follow the public contracts documented by current `verl`, `slime`,
and `alibaba/ROLL` main branches. Contract tests use dependency-free stand-ins
for framework objects, so installing VerifierFuzz does not install an RL stack.

All live wrappers preserve the original return object. Reference evaluation
runs on a bounded background queue; a full queue drops audit samples. Reference
timeouts and adapter conversion failures are counted in `ShadowStats` and do
not raise into the training reward path.

The CLI accepts `--adapter verl`, `trl`, `slime`, and `slime-group`. ROLL
workers require framework-specific construction, so CLI use exports a
preconfigured `RollVerifier` instance and selects `--adapter roll`.

## Corpus format

Audit input is JSONL:

```json
{"case_id":"math-42","prompt":"What is 6 × 7?","completion":"42","reference":"42","metadata":{"data_source":"demo"}}
```

Regression rows wrap the case and specify the expected disagreement. Use
`null` for expected agreement:

```json
{"case":{"case_id":"fixed","completion":"42","reference":"42"},"expected_kind":null}
```

Finding artifacts include the normalized target and reference outcomes,
mutation name and seed, relation type, verifier errors, and minimized
completion.

## Python API

```python
from verifierfuzz import ScorePolicy, VerifierCase, audit_cases
from verifierfuzz.integrations import CallableVerifier
from verifierfuzz.mutators import TextMutationSuite

target = CallableVerifier(
    lambda case: case.reference in case.completion,
    policy=ScorePolicy.zero_one(),
)
reference = CallableVerifier(
    lambda case: case.completion.strip() == case.reference,
    policy=ScorePolicy.zero_one(),
)
findings = audit_cases(
    [VerifierCase(case_id="one", completion="42", reference="42")],
    target,
    reference,
    mutators=[TextMutationSuite()],
    minimize=True,
)
```

For stochastic model judges, use `audit_consistency_async`. Inconsistency
findings report repeated-run counts, pass rate, and standard error; they do not
claim an absolute correctness bug without an independent oracle.

Run the tests:

```bash
python3 -m unittest discover -s tests
```

## Safety and privacy

VerifierFuzz does not attack third-party systems. It audits verifier functions
supplied by the user. Vulnerabilities found in open-source projects should be
reported privately first and published only after maintainers have had a
reasonable opportunity to fix them.

Shadow findings may contain prompts, completions, and raw verifier responses.
Use a private sink or redact fields before exporting artifacts from a training
environment.

## Roadmap

- [x] Framework-independent verifier protocol
- [x] Differential, metamorphic, and stochastic consistency audits
- [x] Hierarchical reducer for textual counterexamples
- [x] verl, slime, ROLL, and TRL reward adapters
- [x] JSONL and SARIF reports
- [ ] JSON tool-call and schema verifier adapter
- [ ] Code-grader integrity and hidden-test checks
- [ ] HTML report and public robustness leaderboard
- [ ] Public, versioned verifier robustness suite

## Contributing

The most valuable early contributions are small real-world verifier examples,
independent reference oracles, and minimized false-positive cases. Please open
an issue before implementing a large adapter.

## License

MIT

## Citation

```bibtex
@software{wang2026verifierfuzz,
  author = {Wang, Xiaofeng},
  title = {VerifierFuzz: Find Reward Exploits Before Your Model Learns Them},
  year = {2026},
  url = {https://github.com/XFWang522/VerifierFuzz}
}
```
