"""Simple, model-agnostic baselines for AMT-MemoryBench selection."""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from datasets.amt_memorybench.schema import MemoryBenchExample, MemoryBenchMemory


class MemorySelector(Protocol):
    def select(
        self, query: str, memories: Sequence[MemoryBenchMemory], k: int
    ) -> tuple[str, ...]:
        """Return up to k selected memory IDs."""


class LexicalMemorySelector:
    """Transparent token-overlap baseline; not a learned retrieval system."""

    def select(
        self, query: str, memories: Sequence[MemoryBenchMemory], k: int
    ) -> tuple[str, ...]:
        query_tokens = _tokens(query)
        ranked = sorted(
            memories,
            key=lambda memory: (-len(query_tokens & _tokens(memory.text)), memory.memory_id),
        )
        return tuple(memory.memory_id for memory in ranked[:k])


class NoMemorySelector:
    """Control that makes no memory available to the answerer."""

    def select(
        self, query: str, memories: Sequence[MemoryBenchMemory], k: int
    ) -> tuple[str, ...]:
        return ()


class RandomMemorySelector:
    """Deterministic random-memory control for a fixed evaluation seed."""

    def __init__(self, seed: int = 17) -> None:
        self._randomizer = random.Random(seed)

    def select(
        self, query: str, memories: Sequence[MemoryBenchMemory], k: int
    ) -> tuple[str, ...]:
        if k < 1:
            return ()
        count = min(k, len(memories))
        return tuple(
            memory.memory_id
            for memory in self._randomizer.sample(list(memories), count)
        )


@dataclass(frozen=True, slots=True)
class SelectionMetrics:
    examples: int
    recall_at_k: float
    exact_required_rate: float


def evaluate_selection(
    examples: Iterable[MemoryBenchExample],
    selector: MemorySelector,
    *,
    k: int = 1,
) -> SelectionMetrics:
    if k < 1:
        raise ValueError("k must be positive")

    count = 0
    recall = 0.0
    exact = 0
    for example in examples:
        selected = set(selector.select(example.query, example.memories, k))
        required = set(example.required_memory_ids)
        recall += len(selected & required) / len(required)
        exact += selected == required
        count += 1
    if not count:
        raise ValueError("cannot evaluate an empty benchmark")
    return SelectionMetrics(
        examples=count,
        recall_at_k=recall / count,
        exact_required_rate=exact / count,
    )


def read_jsonl(path: Path) -> tuple[MemoryBenchExample, ...]:
    examples: list[MemoryBenchExample] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                value = json.loads(line)
                examples.append(_example_from_dict(value["example"]))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid benchmark record on line {line_number}") from error
    return tuple(examples)


def _example_from_dict(value: dict[str, object]) -> MemoryBenchExample:
    raw_memories = value["memories"]
    if not isinstance(raw_memories, list):
        raise TypeError("memories must be a list")
    memories = tuple(
        MemoryBenchMemory(
            memory_id=str(memory["memory_id"]),
            text=str(memory["text"]),
            type=str(memory["type"]),
            scope=str(memory["scope"]),
            complexity=int(memory["complexity"]),
        )
        for memory in raw_memories
    )
    return MemoryBenchExample(
        example_id=str(value["example_id"]),
        user_id=str(value["user_id"]),
        memories=memories,
        query=str(value["query"]),
        answer=str(value["answer"]),
        task_type=value["task_type"],  # type: ignore[arg-type]
        relevant_memory_ids=tuple(value["relevant_memory_ids"]),  # type: ignore[arg-type]
        required_memory_ids=tuple(value["required_memory_ids"]),  # type: ignore[arg-type]
    )


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))