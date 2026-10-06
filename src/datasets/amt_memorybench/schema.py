"""Typed records for the controlled AMT memory benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

TaskType = Literal[
    "single_recall",
    "multi_memory",
    "random_distractor",
    "semantic_distractor",
    "conditional_preference",
]


@dataclass(frozen=True, slots=True)
class MemoryBenchMemory:
    memory_id: str
    text: str
    type: str
    scope: str
    complexity: int

    def __post_init__(self) -> None:
        if not all((self.memory_id, self.text, self.type, self.scope)):
            raise ValueError("memory fields must be non-empty")
        if self.complexity < 1:
            raise ValueError("memory complexity must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MemoryBenchExample:
    example_id: str
    user_id: str
    memories: tuple[MemoryBenchMemory, ...]
    query: str
    answer: str
    task_type: TaskType
    relevant_memory_ids: tuple[str, ...]
    required_memory_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        available = {memory.memory_id for memory in self.memories}
        for ids, field in (
            (self.relevant_memory_ids, "relevant_memory_ids"),
            (self.required_memory_ids, "required_memory_ids"),
        ):
            if not ids or not set(ids).issubset(available):
                raise ValueError(f"{field} must reference available memories")
        if not set(self.required_memory_ids).issubset(self.relevant_memory_ids):
            raise ValueError("required memories must be relevant memories")
        if not self.query or not self.answer:
            raise ValueError("query and answer must be non-empty")

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["memories"] = [memory.to_dict() for memory in self.memories]
        return value