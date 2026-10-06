"""Typed representation of the official LongMemEval JSON records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

Role = Literal["user", "assistant"]


@dataclass(frozen=True, slots=True)
class Turn:
    role: Role
    content: str
    has_answer: bool = False

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Turn:
        role = value.get("role")
        if role not in ("user", "assistant"):
            raise ValueError(f"invalid turn role: {role!r}")
        content = value.get("content")
        if not isinstance(content, str) or not content:
            raise ValueError("turn content must be a non-empty string")
        has_answer = value.get("has_answer", False)
        if not isinstance(has_answer, bool):
            raise ValueError("turn has_answer must be a boolean")
        return cls(role=role, content=content, has_answer=has_answer)


@dataclass(frozen=True, slots=True)
class Session:
    session_id: str
    date: str
    turns: tuple[Turn, ...]


@dataclass(frozen=True, slots=True)
class LongMemEvalRecord:
    question_id: str
    question_type: str
    question: str
    answer: str | int
    question_date: str
    sessions: tuple[Session, ...]
    answer_session_ids: tuple[str, ...]

    @property
    def is_abstention(self) -> bool:
        return self.question_id.endswith("_abs")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> LongMemEvalRecord:
        session_ids = _string_list(value, "haystack_session_ids")
        dates = _string_list(value, "haystack_dates")
        raw_sessions = value.get("haystack_sessions")
        if not isinstance(raw_sessions, list):
            raise ValueError("haystack_sessions must be a list")
        if not len(session_ids) == len(dates) == len(raw_sessions):
            raise ValueError(
                "haystack_session_ids, haystack_dates, and haystack_sessions "
                "must have equal lengths"
            )

        sessions: list[Session] = []
        for session_id, date, raw_turns in zip(session_ids, dates, raw_sessions):
            if not isinstance(raw_turns, list) or not raw_turns:
                raise ValueError("each haystack session must be a non-empty list")
            sessions.append(
                Session(
                    session_id=session_id,
                    date=date,
                    turns=tuple(Turn.from_dict(turn) for turn in raw_turns),
                )
            )

        return cls(
            question_id=_string(value, "question_id"),
            question_type=_string(value, "question_type"),
            question=_string(value, "question"),
            answer=_answer(value),
            question_date=_string(value, "question_date"),
            sessions=tuple(sessions),
            answer_session_ids=tuple(_string_list(value, "answer_session_ids")),
        )


def _string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ValueError(f"{key} must be a non-empty string")
    return item


def _answer(value: Mapping[str, Any]) -> str | int:
    answer = value.get("answer")
    if isinstance(answer, bool) or not isinstance(answer, (str, int)):
        raise ValueError("answer must be a non-empty string or an integer")
    if isinstance(answer, str) and not answer:
        raise ValueError("answer must be a non-empty string or an integer")
    return answer


def _string_list(value: Mapping[str, Any], key: str) -> list[str]:
    items = value.get(key)
    if not isinstance(items, list) or not all(
        isinstance(item, str) and item for item in items
    ):
        raise ValueError(f"{key} must be a list of non-empty strings")
    return items