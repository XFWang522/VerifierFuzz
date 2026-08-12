import unittest

from verifierfuzz.compat import (
    UPSTREAM_CONTRACTS,
    check_contract_source,
    check_upstream_contracts,
)


SOURCES = {
    "verl": """
import inspect
async def _call_with_kwargs_async(raw_fn, extra_kwargs, *args, **kwargs):
    return await raw_fn(*args, **kwargs)
def get_custom_reward_fn(config):
    reward_fn_config = config.reward.get("custom_reward_function")
    reward_kwargs = reward_fn_config.get("reward_kwargs")
    return inspect.iscoroutinefunction(_call_with_kwargs_async)
""",
    "slime": """
async def async_rm(args, sample, **kwargs):
    return 1
async def batched_async_rm(args, samples, **kwargs):
    return [1 for sample in samples]
""",
    "roll": """
class MathRuleRewardWorker:
    def compute_rewards(self, data):
        return data
""",
}


class CompatibilityTests(unittest.TestCase):
    def test_current_contract_shapes_pass(self):
        for framework, source in SOURCES.items():
            with self.subTest(framework=framework):
                self.assertTrue(check_contract_source(framework, source))

    def test_slime_signature_drift_is_actionable(self):
        source = """
async def async_rm(sample):
    return 1
async def batched_async_rm(args, samples, **kwargs):
    return []
"""

        with self.assertRaisesRegex(
            ValueError,
            "async_rm positional arguments changed",
        ):
            check_contract_source("slime", source)

    def test_upstream_probe_reports_each_framework_without_executing_source(self):
        by_url = {
            contract.url: SOURCES[contract.framework]
            for contract in UPSTREAM_CONTRACTS
        }

        results = check_upstream_contracts(fetcher=by_url.__getitem__)

        self.assertEqual(
            [result.framework for result in results],
            ["verl", "slime", "roll"],
        )
        self.assertTrue(all(result.passed for result in results))

    def test_fetch_or_contract_failure_is_reported(self):
        results = check_upstream_contracts(
            fetcher=lambda url: "def unrelated(): pass",
            contracts=[UPSTREAM_CONTRACTS[0]],
        )

        self.assertFalse(results[0].passed)
        self.assertIn("missing function", results[0].error)


if __name__ == "__main__":
    unittest.main()
