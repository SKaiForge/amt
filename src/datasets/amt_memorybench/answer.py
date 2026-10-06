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
    for example in examples:
        selected_ids = selector.select(example.query, example.memories, k)
        selected = tuple(
            memory
            for memory in example.memories
            if memory.memory_id in selected_ids
        )
        prediction = answerer.answer(example.query, selected)
        correct += _normalize(prediction) == _normalize(example.answer)
        count += 1
    if not count:
        raise ValueError("cannot evaluate an empty benchmark")
    return AnswerMetrics(examples=count, exact_accuracy=correct / count)


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))