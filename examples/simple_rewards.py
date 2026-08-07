"""Intentionally weak target and strict reference for CLI examples."""


def target(completion, reference):
    return str(reference) in completion


def oracle(completion, reference):
    return completion.strip() == str(reference).strip()
