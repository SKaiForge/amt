"""Leakage-safe postulate extraction for LongMemEval-Oracle."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from amt_memory.postulates.extractor import PostulateExtractor
from amt_memory.postulates.schema import (
    ExtractionSession,
    ExtractionTurn,
    MemoryPostulate,
    PostulateSource,
)
from datasets.longmemeval.schema import LongMemEvalRecord


def select_stratified_records(
    records: Iterable[LongMemEvalRecord], *, per_type: int, seed: int = 17
) -> tuple[LongMemEvalRecord, ...]:
    """Select a deterministic, balanced non-abstention pilot."""
    if per_type < 1:
        raise ValueError("per_type must be positive")

    grouped: dict[str, list[LongMemEvalRecord]] = defaultdict(list)
    for record in records:
        if not record.is_abstention:
            grouped[record.question_type].append(record)

    randomizer = random.Random(seed)
    selected: list[LongMemEvalRecord] = []
    for question_type in sorted(grouped):
        group = grouped[question_type]
        selected.extend(randomizer.sample(group, min(per_type, len(group))))
    return tuple(selected)


def oracle_extraction_sessions(
    record: LongMemEvalRecord,
) -> tuple[ExtractionSession, ...]:
    """Select oracle evidence turns and remove all benchmark labels."""
    sessions: list[ExtractionSession] = []
    for session in record.sessions:
        turns = tuple(
            ExtractionTurn(index=index, role=turn.role, content=turn.content)
            for index, turn in enumerate(session.turns)
            if turn.has_answer
        )
        if turns:
            sessions.append(
                ExtractionSession(
                    session_id=session.session_id,
                    date=session.date,
                    turns=turns,
                )
            )
    return tuple(sessions)


def all_extraction_sessions(
    record: LongMemEvalRecord,
) -> tuple[ExtractionSession, ...]:
    """Remove benchmark labels while retaining every conversation turn."""
    return tuple(
        ExtractionSession(
            session_id=session.session_id,
            date=session.date,
            turns=tuple(
                ExtractionTurn(index=index, role=turn.role, content=turn.content)
                for index, turn in enumerate(session.turns)
            ),
        )
        for session in record.sessions
    )


def extract_record(
    record: LongMemEvalRecord,
    extractor: PostulateExtractor,
    *,
    evidence_only: bool = True,
) -> tuple[MemoryPostulate, ...]:
    """Extract postulates without exposing question or answer content."""
    postulates: list[MemoryPostulate] = []
    seen_ids: set[str] = set()
    sessions = (
        oracle_extraction_sessions(record)
        if evidence_only
        else all_extraction_sessions(record)
    )
    for session in sessions:
        for candidate in extractor.extract(session):
            postulate_id = _postulate_id(
                record.question_id,
                session.session_id,
                candidate.turn_indexes,
                candidate.canonical_text,
            )
            if postulate_id in seen_ids:
                continue
            seen_ids.add(postulate_id)
            postulates.append(
                MemoryPostulate(
                    postulate_id=postulate_id,
                    subject=candidate.subject,
                    predicate=candidate.predicate,
                    value=candidate.value,
                    type=candidate.type,
                    polarity=candidate.polarity,
                    confidence=candidate.confidence,
                    canonical_text=candidate.canonical_text,
                    source=PostulateSource(
                        question_id=record.question_id,
                        session_id=session.session_id,
                        turn_indexes=candidate.turn_indexes,
                        date=session.date,
                        asserted_by=_assertion_role(session, candidate.turn_indexes),
                    ),
                )
            )
    return tuple(postulates)


def write_pilot(
    records: Iterable[LongMemEvalRecord],
    output_path: Path,
    extractor: PostulateExtractor,
    *,
    per_type: int,
    seed: int = 17,
    evidence_only: bool = True,
) -> tuple[int, int]:
    """Write one JSON object per postulate and return record/postulate counts."""
    selected = select_stratified_records(records, per_type=per_type, seed=seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    postulate_count = 0
    try:
        with temporary.open("w", encoding="utf-8") as output:
            for record in selected:
                for postulate in extract_record(
                    record, extractor, evidence_only=evidence_only
                ):
                    payload = {
                        "schema_version": 2,
                        "extraction_method": extractor.name,
                        "question_type": record.question_type,
                        "postulate": postulate.to_dict(),
                    }
                    output.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    postulate_count += 1
        temporary.replace(output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return len(selected), postulate_count


def _assertion_role(
    session: ExtractionSession, turn_indexes: tuple[int, ...]
) -> str:
    roles = {turn.role for turn in session.turns if turn.index in turn_indexes}
    if len(roles) != 1:
        raise ValueError("a postulate must have exactly one assertion role")
    return roles.pop()


def _postulate_id(
    question_id: str,
    session_id: str,
    turn_indexes: tuple[int, ...],
    canonical_text: str,
) -> str:
    source = "\x1f".join(
        (question_id, session_id, ",".join(map(str, turn_indexes)), canonical_text)
    )
    return f"mem_{hashlib.sha256(source.encode()).hexdigest()[:16]}"