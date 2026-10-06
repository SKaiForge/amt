import json
import tempfile
import unittest
from pathlib import Path

from datasets.amt_memorybench.generate import (
    AMT100_CONFIG,
    SMOKE_CONFIG,
    generate_benchmark,
    write_jsonl,
)


class AMTMemoryBenchTests(unittest.TestCase):
    def test_smoke_has_expected_shape_and_dependencies(self) -> None:
        examples = generate_benchmark(SMOKE_CONFIG)

        self.assertEqual(len(examples), 10 * 5 * 2)
        for example in examples:
            available = {memory.memory_id for memory in example.memories}
            self.assertTrue(set(example.required_memory_ids) <= available)
            self.assertTrue(
                set(example.required_memory_ids) <= set(example.relevant_memory_ids)
            )

    def test_profiles_have_unique_memory_texts(self) -> None:
        examples = generate_benchmark(SMOKE_CONFIG)

        for example in examples:
            texts = [memory.text for memory in example.memories]
            self.assertEqual(len(texts), len(set(texts)))

    def test_queries_and_answers_vary(self) -> None:
        examples = generate_benchmark(SMOKE_CONFIG)
        single_recall = [
            example for example in examples if example.task_type == "single_recall"
        ]

        self.assertGreater(len({example.query for example in single_recall}), 1)
        self.assertGreater(len({example.answer for example in single_recall}), 1)

    def test_generation_is_deterministic(self) -> None:
        self.assertEqual(generate_benchmark(SMOKE_CONFIG), generate_benchmark(SMOKE_CONFIG))

    def test_amt100_dimensions_are_explicit(self) -> None:
        self.assertEqual(AMT100_CONFIG.users, 100)
        self.assertEqual(AMT100_CONFIG.memories_per_user, 50)
        self.assertEqual(AMT100_CONFIG.queries_per_task, 20)

    def test_jsonl_writer_records_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "smoke.jsonl"
            count = write_jsonl(
                generate_benchmark(SMOKE_CONFIG),
                path,
                seed=SMOKE_CONFIG.seed,
                name=SMOKE_CONFIG.name,
            )
            first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(count, 100)
        self.assertEqual(first["schema_version"], 1)
        self.assertIn("relevant_memory_ids", first["example"])


if __name__ == "__main__":
    unittest.main()