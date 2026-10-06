import json
import tempfile
import unittest
from pathlib import Path

from amt_memory.postulates.extractor import (
    DeterministicSemanticExtractor,
    SentenceCandidateExtractor,
)
from amt_memory.postulates.schema import ExtractionSession, ExtractionTurn
from datasets.longmemeval.postulates import (
    all_extraction_sessions,
    extract_record,
    oracle_extraction_sessions,
    select_stratified_records,
    write_pilot,
)
from datasets.longmemeval.schema import LongMemEvalRecord


def _record(question_id: str = "q1", question_type: str = "single-session-user"):
    return LongMemEvalRecord.from_dict(
        {
            "question_id": question_id,
            "question_type": question_type,
            "question": "SECRET BENCHMARK QUESTION",
            "answer": "SECRET BENCHMARK ANSWER",
            "question_date": "2024-01-02",
            "haystack_session_ids": ["s1"],
            "haystack_dates": ["2024-01-01"],
            "haystack_sessions": [
                [
                    {"role": "user", "content": "Ignore this distractor."},
                    {
                        "role": "user",
                        "content": "I prefer tea. I drink it daily.",
                        "has_answer": True,
                    },
                ]
            ],
            "answer_session_ids": ["s1"],
        }
    )


class LongMemEvalPostulateTests(unittest.TestCase):
    def test_oracle_adapter_strips_labels_and_non_evidence_turns(self) -> None:
        sessions = oracle_extraction_sessions(_record())

        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].turns[0].index, 1)
        self.assertNotIn("has_answer", sessions[0].turns[0].__slots__)

    def test_sentence_baseline_preserves_source_provenance(self) -> None:
        postulates = extract_record(_record(), SentenceCandidateExtractor())

        self.assertEqual(len(postulates), 2)
        self.assertEqual(postulates[0].canonical_text, "I prefer tea.")
        self.assertEqual(postulates[0].source.turn_indexes, (1,))
        self.assertEqual(postulates[0].source.asserted_by, "user")

    def test_all_turn_adapter_strips_labels(self) -> None:
        sessions = all_extraction_sessions(_record())

        self.assertEqual(len(sessions[0].turns), 2)
        self.assertNotIn("has_answer", sessions[0].turns[0].__slots__)

    def test_offline_semantic_extractor_types_durable_memories(self) -> None:
        extractor = DeterministicSemanticExtractor()
        record = _record()
        postulates = extract_record(record, extractor, evidence_only=False)

        preferences = [item for item in postulates if item.type == "preference"]
        self.assertEqual(len(preferences), 1)
        self.assertEqual(preferences[0].predicate, "prefers")
        self.assertEqual(preferences[0].source.asserted_by, "user")

    def test_offline_semantic_extractor_filters_generic_assistant_prose(self) -> None:
        extractor = DeterministicSemanticExtractor()
        session = ExtractionSession(
            session_id="s1",
            date="2024-01-01",
            turns=(
                ExtractionTurn(
                    index=0,
                    role="assistant",
                    content="Video editing is a broad field with many useful tools.",
                ),
                ExtractionTurn(
                    index=1,
                    role="assistant",
                    content="I recommend Adobe Premiere Pro for this workflow.",
                ),
            ),
        )

        candidates = extractor.extract(session)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].predicate, "recommended")

    def test_pilot_is_stratified_and_does_not_leak_question_or_answer(self) -> None:
        records = [
            _record("q1", "single-session-user"),
            _record("q2", "single-session-user"),
            _record("q3", "knowledge-update"),
            _record("q4_abs", "knowledge-update"),
        ]
        selected = select_stratified_records(records, per_type=1)
        self.assertEqual({record.question_type for record in selected}, {
            "single-session-user",
            "knowledge-update",
        })

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pilot.jsonl"
            record_count, postulate_count = write_pilot(
                records,
                output,
                SentenceCandidateExtractor(),
                per_type=1,
            )
            content = output.read_text(encoding="utf-8")

        self.assertEqual(record_count, 2)
        self.assertEqual(postulate_count, 4)
        self.assertNotIn("SECRET BENCHMARK QUESTION", content)
        self.assertNotIn("SECRET BENCHMARK ANSWER", content)
        for line in content.splitlines():
            self.assertIn("postulate", json.loads(line))


if __name__ == "__main__":
    unittest.main()