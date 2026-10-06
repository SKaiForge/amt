import io
import json
import unittest

from datasets.longmemeval.dataset import _iter_json_array


class StreamingJsonArrayTests(unittest.TestCase):
    def test_streams_records_across_small_chunks(self) -> None:
        values = [{"id": 1, "text": "alpha"}, {"id": 2, "text": "beta"}]
        source = io.StringIO(json.dumps(values))

        self.assertEqual(list(_iter_json_array(source, chunk_size=7)), values)

    def test_rejects_non_array_root(self) -> None:
        with self.assertRaisesRegex(ValueError, "root must be a JSON array"):
            list(_iter_json_array(io.StringIO('{"id": 1}'), chunk_size=4))


if __name__ == "__main__":
    unittest.main()