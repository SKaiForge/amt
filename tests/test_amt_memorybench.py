import json
import tempfile
import unittest
from pathlib import Path

from datasets.amt_memorybench.generate import (
    ATOMIC_SMOKE_CONFIG,
    ATOMIC_SMOKE_V2_CONFIG,
    generate_atomic_benchmark,
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

    def test_atomic_examples_are_balanced_true_paraphrase_pairs(self) -> None:
        examples = generate_atomic_benchmark(ATOMIC_SMOKE_CONFIG)
        groups: dict[str, list] = {}
        for example in examples:
            groups.setdefault(example.paraphrase_group_id, []).append(example)

        self.assertEqual(len(examples), 10 * 6 * 2)
        self.assertEqual({memory.type for example in examples for memory in example.memories}, {
            "fact", "preference", "goal", "constraint", "event", "relationship"
        })
        for group in groups.values():
            self.assertEqual(len(group), 2)
            self.assertEqual(group[0].answer, group[1].answer)
            self.assertEqual(group[0].required_memory_ids, group[1].required_memory_ids)
            self.assertNotEqual(group[0].query, group[1].query)

    def test_atomic_v2_prompts_match_memory_categories(self) -> None:
        examples = generate_atomic_benchmark(ATOMIC_SMOKE_V2_CONFIG)
        preferences = [example for example in examples if "_preference_" in example.example_id]
        constraints = [example for example in examples if "_constraint_" in example.example_id]

        self.assertTrue(all("framework" not in example.query.lower() for example in preferences))
        self.assertTrue(all("hardware" not in example.query.lower() for example in constraints))
        self.assertTrue(all("ml experiments" in example.query.lower() for example in preferences))

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