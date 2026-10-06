import unittest

from amt_memory.postulates.schema import (
    ExtractionSession,
    ExtractionTurn,
    MemoryPostulate,
    PostulateSource,
)


class PostulateSchemaTests(unittest.TestCase):
    def test_extraction_input_contains_only_conversation_data(self) -> None:
        turn = ExtractionTurn(index=0, role="user", content="I prefer tea.")
        session = ExtractionSession(
            session_id="session-1", date="2024-01-01", turns=(turn,)
        )

        self.assertEqual(
            session.__slots__, ("session_id", "date", "turns")
        )
        self.assertEqual(turn.__slots__, ("index", "role", "content"))

    def test_postulate_serializes_with_provenance(self) -> None:
        postulate = MemoryPostulate(
            postulate_id="mem-1",
            subject="user",
            predicate="prefers",
            value="tea",
            type="preference",
            polarity="positive",
            confidence=1.0,
            canonical_text="The user prefers tea.",
            source=PostulateSource(
                question_id="question-1",
                session_id="session-1",
                turn_indexes=(0,),
                date="2024-01-01",
                asserted_by="user",
            ),
        )

        self.assertEqual(postulate.to_dict()["source"]["turn_indexes"], (0,))

    def test_rejects_invalid_confidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "confidence"):
            MemoryPostulate(
                postulate_id="mem-1",
                subject="user",
                predicate="states",
                value="something",
                type="fact",
                polarity="positive",
                confidence=1.1,
                canonical_text="Something.",
                source=PostulateSource(
                    question_id="question-1",
                    session_id="session-1",
                    turn_indexes=(0,),
                    date="2024-01-01",
                ),
            )


if __name__ == "__main__":
    unittest.main()