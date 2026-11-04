"""Text-memory prompting and answer evaluation interfaces."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from datasets.amt_memorybench.schema import MemoryBenchExample, MemoryBenchMemory


class MemoryAnswerer(Protocol):
    def answer(self, query: str, memories: Sequence[MemoryBenchMemory]) -> str:
        """Answer a query using only the supplied memories."""


class TextModel(Protocol):
    def generate(self, prompt: str) -> str:
        """Generate an answer from a fully formatted text prompt."""


class TextMemoryAnswerer:
    """Adapt a text-generation model to the benchmark answerer interface."""

    def __init__(self, model: TextModel) -> None:
        self._model = model

    def answer(self, query: str, memories: Sequence[MemoryBenchMemory]) -> str:
        return self._model.generate(build_text_prompt(query, memories))


class TextMemoryFormatter:
    """Format selected memories as an explicit textual context."""

    def format(self, memories: Iterable[MemoryBenchMemory]) -> str:
        return "\n".join(f"[{memory.memory_id}] {memory.text}" for memory in memories)


def build_text_prompt(
    query: str, memories: Sequence[MemoryBenchMemory]
) -> str:
    """Build the prompt used by a text-memory answerer."""
    context = TextMemoryFormatter().format(memories)
    return (
        "Answer the question using the memory context when relevant. "
        "Do not invent user preferences.\n\n"
        f"Memory context:\n{context or '(no memory available)'}\n\n"
        f"Question: {query}\nAnswer:"
    )


@dataclass(frozen=True, slots=True)
class AnswerMetrics:
    examples: int
    exact_accuracy: float
    unique_postulates: int
    postulate_accuracy: float


def evaluate_answers(
    examples: Iterable[MemoryBenchExample],
    selector: object,
    answerer: MemoryAnswerer,
    *,
    k: int = 1,
) -> AnswerMetrics:
    """Evaluate an answerer after selecting up to k memories per example."""
    if k < 1:
        raise ValueError("k must be positive")
    count = 0
    correct = 0
    postulate_results: dict[tuple[str, ...], list[bool]] = {}
    for example in examples:
        select_example = getattr(selector, "select_example", None)
        selected_ids = (
            select_example(example)
            if select_example is not None
            else selector.select(example.query, example.memories, k)
        )
        selected = tuple(
            memory
            for memory in example.memories
            if memory.memory_id in selected_ids
        )
        prediction = answerer.answer(example.query, selected)
        is_correct = _normalize(prediction) == _normalize(example.answer)
        correct += is_correct
        required_texts = tuple(
            sorted(
                memory.text
                for memory in example.memories
                if memory.memory_id in example.required_memory_ids
            )
        )
        postulate_results.setdefault(required_texts, []).append(is_correct)
        count += 1
    if not count:
        raise ValueError("cannot evaluate an empty benchmark")
    postulate_accuracy = sum(
        sum(outcomes) / len(outcomes) for outcomes in postulate_results.values()
    ) / len(postulate_results)
    return AnswerMetrics(
        examples=count,
        exact_accuracy=correct / count,
        unique_postulates=len(postulate_results),
        postulate_accuracy=postulate_accuracy,
    )


def _normalize(value: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", value.lower())
    if tokens and tokens[0] in {"a", "an", "the"}:
        tokens = tokens[1:]
    return " ".join(tokens)