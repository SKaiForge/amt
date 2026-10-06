from copy import deepcopy
import unittest

from datasets.longmemeval.schema import LongMemEvalRecord


def _record() -> dict:
    return {
        "question_id": "sample_001",
        "question_type": "single-session-user",
        "question": "What drink do I prefer?",
        "answer": "Green tea",
        "question_date": "2024/01/10 (Wed) 10:00",
        "haystack_session_ids": ["session_001"],
        "haystack_dates": ["2024/01/01 (Mon) 09:00"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "I prefer green tea.", "has_answer": True},
                {"role": "assistant", "content": "I'll remember that."},
            ]
        ],
        "answer_session_ids": ["session_001"],
    }


class LongMemEvalSchemaTests(unittest.TestCase):
    def test_parses_official_record_shape(self) -> None:
        record = LongMemEvalRecord.from_dict(_record())

        self.assertTrue(record.sessions[0].turns[0].has_answer)
        self.assertFalse(record.sessions[0].turns[1].has_answer)
        self.assertFalse(record.is_abstention)

    def test_rejects_misaligned_session_arrays(self) -> None:
        value = deepcopy(_record())
        value["haystack_dates"] = []

        with self.assertRaisesRegex(ValueError, "equal lengths"):
            LongMemEvalRecord.from_dict(value)

    def test_rejects_invalid_turn_role(self) -> None:
        value = deepcopy(_record())
        value["haystack_sessions"][0][0]["role"] = "system"

        with self.assertRaisesRegex(ValueError, "invalid turn role"):
            LongMemEvalRecord.from_dict(value)

    def test_preserves_integer_answer(self) -> None:
        value = _record()
        value["answer"] = 2

        record = LongMemEvalRecord.from_dict(value)

        self.assertEqual(record.answer, 2)


if __name__ == "__main__":
    unittest.main()