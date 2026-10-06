"""Deterministic generation of AMT-MemoryBench examples."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from datasets.amt_memorybench.schema import (
    MemoryBenchExample,
    MemoryBenchMemory,
    TaskType,
)

_PROFILE_FACTS = (
    tuple(
        ("preference", "response_style", f"The user prefers {value}.", value)
        for value in (
            "concise answers",
            "technical explanations",
            "step-by-step explanations",
            "short code samples",
            "examples before theory",
        )
    )
    + tuple(
        ("preference", "programming", f"The user prefers {language} examples.", f"{language} examples")
        for language in ("Python", "Rust", "JavaScript", "Go", "Java", "C++", "TypeScript", "Kotlin")
    )
    + tuple(
        ("preference", "ml", f"The user prefers {framework} for ML experiments.", f"{framework} for ML experiments")
        for framework in ("Python", "PyTorch", "JAX", "TensorFlow", "Julia", "scikit-learn")
    )
    + tuple(
        ("preference", "backend", f"The user prefers {language} for production backend systems.", f"{language} for production backend systems")
        for language in ("Rust", "Go", "Java", "C#", "Kotlin")
    )
    + tuple(
        ("preference", "frontend", f"The user prefers {language} for frontend work.", f"{language} for frontend work")
        for language in ("JavaScript", "TypeScript", "Elm", "Svelte")
    )
    + tuple(
        ("project", "project", f"The user is building {project}.", project)
        for project in (
            "an NL2SQL system",
            "an LLM memory system",
            "a recommendation engine",
            "a document search service",
            "a data-quality dashboard",
        )
    )
    + tuple(
        ("goal", "learning", f"The user is learning {subject}.", subject)
        for subject in (
            "reinforcement learning",
            "information retrieval",
            "distributed systems",
            "compiler design",
            "causal inference",
        )
    )
    + tuple(
        ("constraint", "hardware", f"The user prefers solutions that fit on {hardware}.", hardware)
        for hardware in ("one GPU", "a laptop", "16 GB of memory", "a small CPU instance", "an offline environment")
    )
    + tuple(
        ("skill", "skill", f"The user knows {skill}.", skill)
        for skill in ("Python", "SQL", "Docker", "Linux", "statistics", "Git", "linear algebra")
    )
)

_TASK_TYPES: tuple[TaskType, ...] = (
    "single_recall",
    "multi_memory",
    "random_distractor",
    "semantic_distractor",
    "conditional_preference",
)


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    name: str
    users: int
    memories_per_user: int
    queries_per_task: int
    seed: int = 17

    def __post_init__(self) -> None:
        if not self.name or min(self.users, self.memories_per_user, self.queries_per_task) < 1:
            raise ValueError("benchmark name and dimensions must be non-empty and positive")


SMOKE_CONFIG = BenchmarkConfig(
    name="amt-smoke", users=10, memories_per_user=20, queries_per_task=2
)
AMT100_CONFIG = BenchmarkConfig(
    name="amt-100", users=100, memories_per_user=50, queries_per_task=20
)


def generate_benchmark(config: BenchmarkConfig) -> tuple[MemoryBenchExample, ...]:
    """Generate a deterministic benchmark with five static task families."""
    randomizer = random.Random(config.seed)
    examples: list[MemoryBenchExample] = []
    for user_number in range(1, config.users + 1):
        user_id = f"user_{user_number:03d}"
        memories = _user_memories(user_id, config.memories_per_user, randomizer)
        for task_type in _TASK_TYPES:
            for query_number in range(config.queries_per_task):
                examples.append(
                    _example(
                        config.name,
                        user_id,
                        memories,
                        task_type,
                        query_number,
                        randomizer,
                    )
                )
    return tuple(examples)


def write_jsonl(
    examples: Iterable[MemoryBenchExample], output_path: Path, *, seed: int, name: str
) -> int:
    """Write examples atomically and return the number of records written."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8") as output:
            for example in examples:
                output.write(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "benchmark": name,
                            "seed": seed,
                            "example": example.to_dict(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                count += 1
        temporary.replace(output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return count


def _user_memories(
    user_id: str, count: int, randomizer: random.Random
) -> tuple[MemoryBenchMemory, ...]:
    facts = list(_PROFILE_FACTS)
    if count > len(facts):
        raise ValueError("requested more memories than the profile catalog contains")
    randomizer.shuffle(facts)
    selected = facts[:count]
    return tuple(
        MemoryBenchMemory(
            memory_id=f"{user_id}_m{index + 1:03d}",
            text=fact[2],
            type=fact[0],
            scope=fact[1],
            complexity=1 if index < count // 2 else 2,
        )
        for index, fact in enumerate(selected)
    )


def _example(
    benchmark: str,
    user_id: str,
    memories: tuple[MemoryBenchMemory, ...],
    task_type: TaskType,
    query_number: int,
    randomizer: random.Random,
) -> MemoryBenchExample:
    ml_memories = _find_memories(memories, "ml")
    primary = ml_memories[query_number % len(ml_memories)]
    if task_type == "single_recall":
        required = (primary,)
        query = (
            "What does the user prefer for ML experiments?"
            if query_number % 2 == 0
            else "Which ML framework matches the user's stated preference?"
        )
        answer = _preference_value(primary)
    elif task_type == "multi_memory":
        styles = _find_memories(memories, "response_style")
        style = styles[query_number % len(styles)]
        required = (primary, style)
        query = "Give an answer about ML experiments using the user's preferred style."
        answer = f"{_preference_value(primary)}; use {_preference_value(style)}."
    elif task_type == "conditional_preference":
        required = (primary,)
        query = (
            "The user is running an ML experiment. Which choice should guide the example?"
            if query_number % 2 == 0
            else "For this ML experiment, what should the example use?"
        )
        answer = _preference_value(primary)
    elif task_type == "semantic_distractor":
        required = (primary,)
        distractor = _find_memory(memories, "backend")
        query = "Recommend a language for the user's machine-learning experiment, not a production backend."
        answer = _preference_value(primary)
        memories = _with_memories(memories, (primary, distractor), randomizer)
    else:
        required = (primary,)
        distractors = tuple(memory for memory in memories if memory != primary)
        query = (
            "What should the user use for an ML experiment?"
            if query_number % 2 == 0
            else "Which stored preference applies to the ML experiment?"
        )
        answer = _preference_value(primary)
        memories = _with_memories(memories, (primary, *distractors), randomizer)

    relevant = tuple(memory.memory_id for memory in required)
    return MemoryBenchExample(
        example_id=f"{benchmark}_{user_id}_{task_type}_{query_number:02d}",
        user_id=user_id,
        memories=memories,
        query=query,
        answer=answer,
        task_type=task_type,
        relevant_memory_ids=relevant,
        required_memory_ids=relevant,
    )


def _find_memory(memories: tuple[MemoryBenchMemory, ...], scope: str) -> MemoryBenchMemory:
    return _find_memories(memories, scope)[0]


def _find_memories(
    memories: tuple[MemoryBenchMemory, ...], scope: str
) -> tuple[MemoryBenchMemory, ...]:
    matches = tuple(memory for memory in memories if memory.scope == scope)
    if not matches:
        raise ValueError(f"generated profile has no memory in scope {scope!r}")
    return matches


def _preference_value(memory: MemoryBenchMemory) -> str:
    prefix = "The user prefers "
    if not memory.text.startswith(prefix):
        raise ValueError(f"memory is not a preference: {memory.memory_id}")
    return memory.text[len(prefix) :].rstrip(".")


def _with_memories(
    memories: tuple[MemoryBenchMemory, ...],
    relevant: tuple[MemoryBenchMemory, ...],
    randomizer: random.Random,
) -> tuple[MemoryBenchMemory, ...]:
    selected = list(memories)
    randomizer.shuffle(selected)
    return tuple(selected)