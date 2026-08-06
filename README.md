# VerifierFuzz

**Find reward exploits before your model learns them.**

VerifierFuzz is an early-stage property-based testing toolkit for RLVR reward
verifiers. It searches for completions that a target verifier rewards even
though a stricter reference rejects them, then turns every disagreement into a
reproducible regression case.

> Status: project scaffold and executable proof of concept. The public API will
> change while the first real verifier adapters are developed.

## Why

In reinforcement learning with verifiable rewards, the verifier is part of the
training objective. A false positive is not merely an evaluation bug: repeated
optimization can teach the policy to exploit it.

VerifierFuzz aims to provide:

- domain-aware mutation for math answers, tool calls, and code graders;
- differential checking against independent reference oracles;
- minimization of reward-positive incorrect completions;
- versioned regression corpora and CI reports;
- adapters for common post-training and evaluation frameworks.

## Quick start

The initial proof of concept has no runtime dependencies:

```bash
python3 -m verifierfuzz demo
```

It audits an intentionally weak math verifier and prints candidates that receive
reward from the weak verifier but fail the strict answer contract.

Run the tests:

```bash
python3 -m unittest discover -s tests
```

## Design principle

VerifierFuzz does not attack third-party systems. It audits verifier functions
supplied by the user. Vulnerabilities found in open-source projects should be
reported privately first and published only after maintainers have had a
reasonable opportunity to fix them.

## Roadmap

- [ ] Math final-answer verifier adapter and metamorphic mutators
- [ ] Hierarchical reducer for minimal counterexamples
- [ ] JSON tool-call and schema verifier adapter
- [ ] Code-grader integrity and hidden-test checks
- [ ] SARIF and HTML reports
- [ ] Public, versioned verifier robustness suite

## Contributing

The most valuable early contributions are small real-world verifier examples,
independent reference oracles, and minimized false-positive cases. Please open
an issue before implementing a large adapter.

## License

MIT
