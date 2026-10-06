import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import call, patch

from datasets.longmemeval.cli import main


class LongMemEvalCliTests(unittest.TestCase):
    @patch("datasets.longmemeval.cli.download")
    def test_download_all_requests_every_variant(self, mock_download) -> None:
        mock_download.side_effect = lambda variant, data_dir, force: Path(
            f"{data_dir}/{variant}.json"
        )

        with redirect_stdout(io.StringIO()):
            result = main(["download", "all", "--data-dir", "target"])

        self.assertEqual(result, 0)
        self.assertEqual(
            mock_download.call_args_list,
            [
                call("oracle", Path("target"), force=False),
                call("s", Path("target"), force=False),
                call("m", Path("target"), force=False),
            ],
        )


if __name__ == "__main__":
    unittest.main()