"""Contracts for leakage-safe atomic memory extraction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

PostulateType = Literal[
    "fact",
    "preference",
    "goal",
    "constraint",
    "event",
    "relationship",
    "temporal_state",
]
Polarity = Literal["positive", "negative"]
Role = Literal["user", "assistant"]


@dataclass(frozen=True, slots=True)
class ExtractionTurn:
    """A turn visible to an extractor; benchmark labels are intentionally absent."""

    index: int
    role: Role
    content: str

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("turn index must be non-negative")
        if self.role not in ("user", "assistant"):
            raise ValueError(f"invalid turn role: {self.role!r}")
        if not self.content:
            raise ValueError("turn content must be non-empty")


@dataclass(frozen=True, slots=True)
class ExtractionSession:
    """The complete information an extractor may consume."""

    session_id: str
    date: str
    turns: tuple[ExtractionTurn, ...]

    def __post_init__(self) -> None:
        if not self.session_id or not self.date:
            raise ValueError("session id and date must be non-empty")
        if not self.turns:
            raise ValueError("extraction session must contain at least one turn")


@dataclass(frozen=True, slots=True)
class PostulateSource:
    question_id: str
    session_id: str
    turn_indexes: tuple[int, ...]
    date: str
    asserted_by: Role | None = None

    def __post_init__(self) -> None:
        if not self.question_id or not self.session_id or not self.date:
            raise ValueError("postulate source identifiers must be non-empty")
        if not self.turn_indexes or any(index < 0 for index in self.turn_indexes):
            raise ValueError("postulate source requires non-negative turn indexes")
        if self.asserted_by not in (None, "user", "assistant"):
            raise ValueError(f"invalid assertion role: {self.asserted_by!r}")


@dataclass(frozen=True, slots=True)
class MemoryPostulate:
    postulate_id: str
    subject: str
    predicate: str
    value: str
    type: PostulateType
    polarity: Polarity
    confidence: float
    canonical_text: str
    source: PostulateSource
    valid_from: str | None = None
    valid_until: str | None = None
    supersedes: str | None = None

    def __post_init__(self) -> None:
        required = (
            self.postulate_id,
            self.subject,
            self.predicate,
            self.value,
            self.canonical_text,
        )
        if any(not item for item in required):
            raise ValueError("postulate text fields must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("postulate confidence must be between 0 and 1")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)