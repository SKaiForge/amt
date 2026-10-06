"""Command line access to LongMemEval data."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Sequence

from datasets.longmemeval.dataset import (
    DATASET_FILES,
    dataset_path,
    download,
    iter_records,
)
from datasets.longmemeval.postulates import write_pilot
from amt_memory.postulates.extractor import (
    DeterministicSemanticExtractor,
    SentenceCandidateExtractor,
)

DEFAULT_DATA_DIR = Path("data/raw/longmemeval")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="longmemeval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download")
    _variant_argument(download_parser, include_all=True)
    download_parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    download_parser.add_argument("--force", action="store_true")

    inspect_parser = subparsers.add_parser("inspect")
    _variant_argument(inspect_parser)
    inspect_parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    inspect_parser.add_argument("--limit", type=int, default=5)

    extract_parser = subparsers.add_parser("extract-pilot")
    extract_parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    extract_parser.add_argument(
        "--output",
        type=Path,
    )
    extract_parser.add_argument(
        "--extractor",
        choices=("sentence", "offline-semantic"),
        default="sentence",
    )
    extract_parser.add_argument("--per-type", type=int, default=8)
    extract_parser.add_argument("--seed", type=int, default=17)

    args = parser.parse_args(argv)
    if args.command == "download":
        variants = DATASET_FILES if args.variant == "all" else (args.variant,)
        for variant in variants:
            path = download(variant, args.data_dir, force=args.force)
            print(path)
        return 0

    if args.command == "extract-pilot":
        input_path = dataset_path("oracle", args.data_dir)
        semantic = args.extractor == "offline-semantic"
        extractor = (
            DeterministicSemanticExtractor()
            if semantic
            else SentenceCandidateExtractor()
        )
        output_path = args.output or Path(
            "data/processed/longmemeval/"
            + (
                "oracle_postulates_semantic.jsonl"
                if semantic
                else "oracle_postulates.jsonl"
            )
        )
        records, postulates = write_pilot(
            iter_records(input_path),
            output_path,
            extractor,
            per_type=args.per_type,
            seed=args.seed,
            evidence_only=not semantic,
        )
        print(
            json.dumps(
                {
                    "input": str(input_path),
                    "output": str(output_path),
                    "records": records,
                    "postulates": postulates,
                },
                indent=2,
            )
        )
        return 0

    path = dataset_path(args.variant, args.data_dir)
    records = list(islice(iter_records(path), args.limit))
    question_types = Counter(record.question_type for record in records)
    summary = {
        "file": str(path),
        "records_inspected": len(records),
        "record_fields": [
            "question_id",
            "question_type",
            "question",
            "answer",
            "question_date",
            "haystack_session_ids",
            "haystack_dates",
            "haystack_sessions",
            "answer_session_ids",
        ],
        "turn_fields": ["role", "content", "has_answer (optional)"],
        "question_types_in_sample": dict(sorted(question_types.items())),
        "sample": _sample(records[0]) if records else None,
    }
    print(json.dumps(summary, indent=2))
    return 0


def _variant_argument(
    parser: argparse.ArgumentParser, *, include_all: bool = False
) -> None:
    choices = (*DATASET_FILES, "all") if include_all else tuple(DATASET_FILES)
    parser.add_argument("variant", choices=choices)


def _sample(record: object) -> dict[str, object]:
    first_session = record.sessions[0]
    first_turn = first_session.turns[0]
    return {
        "question_id": record.question_id,
        "question_type": record.question_type,
        "question": record.question,
        "answer": record.answer,
        "question_date": record.question_date,
        "session_count": len(record.sessions),
        "answer_session_ids": record.answer_session_ids,
        "first_session": {
            "session_id": first_session.session_id,
            "date": first_session.date,
            "turn_count": len(first_session.turns),
            "first_turn": {
                "role": first_turn.role,
                "content": first_turn.content,
                "has_answer": first_turn.has_answer,
            },
        },
    }