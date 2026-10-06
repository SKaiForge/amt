"""Extractor interface and dependency-free candidate baseline."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from amt_memory.postulates.schema import ExtractionSession, Polarity, PostulateType


@dataclass(frozen=True, slots=True)
class PostulateCandidate:
    subject: str
    predicate: str
    value: str
    type: PostulateType
    polarity: Polarity
    confidence: float
    canonical_text: str
    turn_indexes: tuple[int, ...]


class PostulateExtractor(Protocol):
    name: str

    def extract(self, session: ExtractionSession) -> tuple[PostulateCandidate, ...]:
        """Extract candidates using conversation data only."""
        ...


class SentenceCandidateExtractor:
    """Low-confidence baseline that treats each non-empty sentence as a claim."""

    name = "sentence-candidate-v1"

    def extract(self, session: ExtractionSession) -> tuple[PostulateCandidate, ...]:
        candidates: list[PostulateCandidate] = []
        for turn in session.turns:
            for sentence in _sentences(turn.content):
                candidates.append(
                    PostulateCandidate(
                        subject=turn.role,
                        predicate="states",
                        value=sentence,
                        type="fact",
                        polarity="positive",
                        confidence=0.25,
                        canonical_text=sentence,
                        turn_indexes=(turn.index,),
                    )
                )
        return tuple(candidates)


class DeterministicSemanticExtractor:
    """Offline rules for selecting and typing durable conversational memories."""

    name = "offline-semantic-v1"

    def extract(self, session: ExtractionSession) -> tuple[PostulateCandidate, ...]:
        candidates: list[PostulateCandidate] = []
        for turn in session.turns:
            for sentence in _sentences(turn.content):
                candidate = _semantic_candidate(turn.index, turn.role, sentence)
                if candidate is not None:
                    candidates.append(candidate)
        return tuple(candidates)


def _sentences(content: str) -> tuple[str, ...]:
    normalized = " ".join(content.split())
    return tuple(
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", normalized)
        if sentence.strip()
    )


_FILLER_PREFIXES = (
    "as an ai",
    "as a large language model",
    "congratulations",
    "great question",
    "i'd be happy to help",
    "i am happy to help",
    "i'm happy to help",
    "of course",
    "sure",
    "thank you",
    "thanks",
    "you're welcome",
)
_GENERIC_ADVICE_PREFIXES = (
    "consider ",
    "here are ",
    "it is important to ",
    "it's important to ",
    "you can ",
    "you could ",
    "you may ",
    "you might ",
    "you should ",
)


def _semantic_candidate(
    turn_index: int, role: str, sentence: str
) -> PostulateCandidate | None:
    text = sentence.strip(" \t-*#")
    lowered = text.lower()
    if (
        len(text.split()) < 3
        or text.endswith("?")
        or lowered.startswith(_FILLER_PREFIXES)
        or lowered.startswith(_GENERIC_ADVICE_PREFIXES)
    ):
        return None

    if role == "assistant":
        recommendation = re.search(
            r"\bI\s+(?:would\s+)?(?:recommend|suggest|advise)\s+(.+)",
            text,
            re.IGNORECASE,
        )
        if recommendation:
            return _candidate(
                turn_index,
                role,
                text,
                "recommended",
                recommendation.group(1).strip(" ."),
                "fact",
                "positive",
                0.65,
            )
        commitment = re.search(
            r"\bI(?:'ll| will)\s+(remember|remind|track|note|save)\s+(.+)",
            text,
            re.IGNORECASE,
        )
        if commitment:
            return _candidate(
                turn_index,
                role,
                text,
                "committed_to",
                commitment.group(2).strip(" ."),
                "goal",
                "positive",
                0.7,
            )
        supplied_information = re.search(
            r"\b(?:you (?:said|mentioned|told me)|I remember (?:that )?)\s+(.+)",
            text,
            re.IGNORECASE,
        )
        if supplied_information:
            return _candidate(
                turn_index,
                role,
                text,
                "recalls",
                supplied_information.group(1).strip(" ."),
                "fact",
                "positive",
                0.7,
            )
        return None

    patterns: tuple[
        tuple[re.Pattern[str], str, PostulateType, Polarity, float], ...
    ] = (
        (
            re.compile(
                r"\bI\s+(?:really\s+)?(prefer|like|love|enjoy)\s+(?:to\s+)?(.+)",
                re.IGNORECASE,
            ),
            "prefers",
            "preference",
            "positive",
            0.8,
        ),
        (
            re.compile(r"\bI\s+(dislike|hate)\s+(.+)", re.IGNORECASE),
            "prefers",
            "preference",
            "negative",
            0.8,
        ),
        (
            re.compile(
                r"\bI(?:'m| am)\s+(?:trying|planning|hoping|intending)\s+"
                r"(?:to|on)\s+(.+)",
                re.IGNORECASE,
            ),
            "intends",
            "goal",
            "positive",
            0.75,
        ),
        (
            re.compile(
                r"\bI\s+(?:still\s+)?(?:need|want|intend)\s+(?:to\s+)?(.+)",
                re.IGNORECASE,
            ),
            "intends",
            "goal",
            "positive",
            0.7,
        ),
        (
            re.compile(r"\bI\s+(?:can't|cannot|must|have to)\s+(.+)", re.IGNORECASE),
            "is_constrained_to",
            "constraint",
            "positive",
            0.75,
        ),
        (
            re.compile(
                r"\b(?:my|our)\s+(wife|husband|partner|friend|mother|father|"
                r"sister|brother|son|daughter|coworker|colleague)\b(.+)",
                re.IGNORECASE,
            ),
            "has_relationship",
            "relationship",
            "positive",
            0.7,
        ),
        (
            re.compile(
                r"\bI\s+(?:just\s+)?(?:got|went|visited|bought|started|finished|"
                r"attended|received|wore|had)\s+(.+)",
                re.IGNORECASE,
            ),
            "experienced",
            "event",
            "positive",
            0.7,
        ),
    )
    for pattern, predicate, postulate_type, polarity, confidence in patterns:
        match = pattern.search(text)
        if match:
            value = match.group(match.lastindex or 0).strip(" .")
            return _candidate(
                turn_index,
                role,
                text,
                predicate,
                value,
                postulate_type,
                polarity,
                confidence,
            )

    first_person = re.search(r"\b(I|I'm|I've|I'd|my|we|our)\b", text, re.IGNORECASE)
    if first_person:
        return _candidate(
            turn_index,
            role,
            text,
            "states",
            text,
            "fact",
            "positive",
            0.5,
        )
    return None


def _candidate(
    turn_index: int,
    role: str,
    text: str,
    predicate: str,
    value: str,
    postulate_type: PostulateType,
    polarity: Polarity,
    confidence: float,
) -> PostulateCandidate:
    return PostulateCandidate(
        subject=role,
        predicate=predicate,
        value=value,
        type=postulate_type,
        polarity=polarity,
        confidence=confidence,
        canonical_text=text,
        turn_indexes=(turn_index,),
    )