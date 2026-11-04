import unittest

from datasets.amt_memorybench.answer import (
    TextMemoryFormatter,
    TextMemoryAnswerer,
    build_text_prompt,
    evaluate_answers,
)
from datasets.amt_memorybench.evaluate import GoldMemorySelector, LexicalMemorySelector
from datasets.amt_memorybench.generate import SMOKE_CONFIG, generate_benchmark
from datasets.amt_memorybench.generate import ATOMIC_SMOKE_V2_CONFIG, generate_atomic_benchmark


class _ExpectedAnswerer:
    def answer(self, query, memories):
        return memories[0].text.removeprefix("The user prefers ").rstrip(".")


class _PromptCheckingModel:
    def generate(self, prompt):
        if "Memory context:" not in prompt:
            raise AssertionError("memory context was not included in prompt")
        return "Python for ML experiments"


class _AnswerLookup:
    def __init__(self, examples):
        self._answers = {example.query: example.answer for example in examples}

    def answer(self, query, memories):
        return self._answers[query]


class AMTMemoryBenchAnswerTests(unittest.TestCase):
    def test_formatter_preserves_memory_ids_and_text(self) -> None:
        example = generate_benchmark(SMOKE_CONFIG)[0]

        formatted = TextMemoryFormatter().format(example.memories[:1])

        self.assertIn(example.memories[0].memory_id, formatted)
        self.assertIn(example.memories[0].text, formatted)

    def test_prompt_explicitly_handles_empty_memory(self) -> None:
        prompt = build_text_prompt("What is preferred?", ())

        self.assertIn("(no memory available)", prompt)
        self.assertIn("What is preferred?", prompt)

    def test_answer_evaluation_uses_selected_memories(self) -> None:
        examples = [
            example
            for example in generate_benchmark(SMOKE_CONFIG)
            if example.task_type == "single_recall"
        ]

        metrics = evaluate_answers(
            examples,
            LexicalMemorySelector(),
            _ExpectedAnswerer(),
        )

        self.assertEqual(metrics.examples, 20)
        self.assertGreater(metrics.unique_postulates, 0)
        self.assertGreaterEqual(metrics.postulate_accuracy, 0.0)
        self.assertGreater(metrics.exact_accuracy, 0.0)

    def test_answer_scoring_normalizes_articles_but_rejects_explanations(self) -> None:
        examples = generate_atomic_benchmark(ATOMIC_SMOKE_V2_CONFIG)
        event_example = next(
            example
            for example in examples
            if example.example_id.endswith("event_00")
        )

        class _VariantAnswerer:
            def answer(self, query, memories):
                answer = event_example.answer.split(" ", 1)[1]
                return f"the {answer}"

        metrics = evaluate_answers(
            [event_example], GoldMemorySelector(), _VariantAnswerer()
        )
        self.assertEqual(metrics.exact_accuracy, 1.0)

        class _VerboseAnswerer:
            def answer(self, query, memories):
                return "The user attended a machine learning workshop yesterday."

        verbose_metrics = evaluate_answers(
            [event_example], GoldMemorySelector(), _VerboseAnswerer()
        )
        self.assertEqual(verbose_metrics.exact_accuracy, 0.0)

    def test_postulate_accuracy_macro_averages_paraphrase_groups(self) -> None:
        examples = generate_atomic_benchmark(ATOMIC_SMOKE_V2_CONFIG)
        pair = [
            example
            for example in examples
            if example.paraphrase_group_id == examples[0].paraphrase_group_id
        ]

        metrics = evaluate_answers(pair, GoldMemorySelector(), _AnswerLookup(pair))

        self.assertEqual(metrics.examples, 2)
        self.assertEqual(metrics.unique_postulates, 1)
        self.assertEqual(metrics.postulate_accuracy, 1.0)

    def test_text_answerer_connects_model_to_prompt_builder(self) -> None:
        example = generate_benchmark(SMOKE_CONFIG)[0]

        answer = TextMemoryAnswerer(_PromptCheckingModel()).answer(
            example.query, example.memories[:1]
        )

        self.assertEqual(answer, "Python for ML experiments")

    def test_gold_selector_exposes_required_memory(self) -> None:
        examples = [
            example
            for example in generate_benchmark(SMOKE_CONFIG)
            if example.task_type == "single_recall"
        ]

        metrics = evaluate_answers(examples, GoldMemorySelector(), _ExpectedAnswerer())

        self.assertEqual(metrics.exact_accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()