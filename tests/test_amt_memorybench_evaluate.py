import tempfile
import unittest
from pathlib import Path

from datasets.amt_memorybench.evaluate import (
    LexicalMemorySelector,
    NoMemorySelector,
    RandomMemorySelector,
    evaluate_selection,
    read_jsonl,
)
from datasets.amt_memorybench.generate import SMOKE_CONFIG, generate_benchmark, write_jsonl
from datasets.amt_memorybench.generate import ATOMIC_SMOKE_CONFIG, generate_atomic_benchmark


class AMTMemoryBenchEvaluationTests(unittest.TestCase):
    def test_lexical_selector_recovers_single_memory_on_smoke(self) -> None:
        examples = [
            example
            for example in generate_benchmark(SMOKE_CONFIG)
            if example.task_type == "single_recall"
        ]

        metrics = evaluate_selection(examples, LexicalMemorySelector())

        self.assertEqual(metrics.examples, 20)
        self.assertGreater(metrics.recall_at_k, 0.0)

    def test_read_jsonl_round_trips_generated_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "smoke.jsonl"
            write_jsonl(
                generate_benchmark(SMOKE_CONFIG),
                path,
                seed=SMOKE_CONFIG.seed,
                name=SMOKE_CONFIG.name,
            )
            examples = read_jsonl(path)

        self.assertEqual(len(examples), 100)
        self.assertEqual(examples[0].example_id, "amt-smoke_user_001_single_recall_00")

    def test_read_jsonl_preserves_atomic_paraphrase_groups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "atomic-smoke.jsonl"
            write_jsonl(
                generate_atomic_benchmark(ATOMIC_SMOKE_CONFIG),
                path,
                seed=ATOMIC_SMOKE_CONFIG.seed,
                name=ATOMIC_SMOKE_CONFIG.name,
            )
            examples = read_jsonl(path)

        groups = {example.paraphrase_group_id for example in examples}
        self.assertEqual(len(examples), 120)
        self.assertEqual(len(groups), 60)
        self.assertNotIn(None, groups)

    def test_no_memory_control_never_recovers_required_memory(self) -> None:
        examples = generate_benchmark(SMOKE_CONFIG)

        metrics = evaluate_selection(examples, NoMemorySelector())

        self.assertEqual(metrics.recall_at_k, 0.0)
        self.assertEqual(metrics.exact_required_rate, 0.0)

    def test_random_selector_is_reproducible(self) -> None:
        examples = generate_benchmark(SMOKE_CONFIG)
        first = evaluate_selection(examples, RandomMemorySelector(seed=3))
        second = evaluate_selection(examples, RandomMemorySelector(seed=3))

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()