import ast
import contextlib
import hashlib
import io
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from datasets.amt_memorybench.generate import (
    ATOMIC_SMOKE_V2_CONFIG,
    generate_atomic_benchmark,
    write_jsonl,
)


NOTEBOOK = Path(__file__).resolve().parents[1] / "amt.ipynb"


def cell_source(anchor):
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    return next("".join(cell["source"]) for cell in cells if anchor in "".join(cell["source"]))


def helper_namespace():
    tree = ast.parse(cell_source("def score_answer"))
    selected = [
        node for node in tree.body
        if isinstance(node, ast.Import) and any(alias.name == "unicodedata" for alias in node.names)
        or isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "APPROVED_ALIASES"
            for target in node.targets
        )
        or isinstance(node, ast.FunctionDef) and node.name in {
            "normalize", "score_answer", "atomic_json", "memory_for"
        }
    ]
    namespace = {"json": json}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "notebook-helpers", "exec"), namespace)
    return namespace


class AMTNotebookTests(unittest.TestCase):
    def test_python_cells_parse(self):
        cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
        for index, cell in enumerate(cells):
            if cell["cell_type"] != "code":
                continue
            source = "\n".join(
                line for line in "".join(cell["source"]).splitlines()
                if not line.lstrip().startswith(("!", "%"))
            )
            with self.subTest(cell=index):
                ast.parse(source)

    def test_scoring_preserves_symbols_and_rejects_verbose_matches(self):
        score = helper_namespace()["score_answer"]
        self.assertEqual(score(" JULIA ", "Julia"), {"strict": True, "alias": True})
        self.assertFalse(score("Morgan. Explanation follows", "Morgan")["alias"])
        self.assertFalse(score("C", "C++")["strict"])
        self.assertFalse(score("16 GB", "16 GB of memory")["strict"])
        self.assertEqual(
            score("the machine learning workshop", "a machine-learning workshop"),
            {"strict": False, "alias": True},
        )

    def test_chat_prompt_and_answer_boundaries(self):
        tree = ast.parse(cell_source("def score_answer"))
        functions = {
            node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        calls = []

        class TokenBatch(list):
            def to(self, device):
                return self

        class ChatTokenizer:
            def apply_chat_template(self, messages, **options):
                calls.append((messages, options))
                return "formatted assistant generation prompt"

            def __call__(self, text, **options):
                calls.append((text, options))
                return SimpleNamespace(input_ids=TokenBatch([[1, 2]]))

        namespace = {
            "tokenizer": ChatTokenizer(), "input_device": "test-device", "ANSWER_STOP_ID": 99,
            "torch": SimpleNamespace(
                long="long", tensor=lambda value, **options: TokenBatch(value),
                cat=lambda values, dim: TokenBatch([values[0][0] + values[1][0]]),
            ),
        }
        exec(compile(ast.Module(body=[functions["prompt_for"], functions["build_answer_ids"]],
                                type_ignores=[]), "chat-boundaries", "exec"), namespace)
        namespace["prompt_for"]("Which language?", "The user knows Java.")
        messages, options = calls[-1]
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertIn("The user knows Java.", messages[1]["content"])
        self.assertEqual(options, {
            "tokenize": False, "add_generation_prompt": True, "enable_thinking": False,
        })
        namespace["prompt_for"]("Which language?")
        self.assertNotIn("Memory:", calls[-1][0][1]["content"])
        answer_ids = namespace["build_answer_ids"]("Java")
        self.assertEqual(calls[-1], ("Java", {"return_tensors": "pt", "add_special_tokens": False}))
        self.assertEqual(answer_ids, [[1, 2, 99]])
        for name in ("generate_from_prefix", "train_one_free_token"):
            tokenizer_calls = [node for node in ast.walk(functions[name])
                               if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                               and node.func.id == "tokenizer"]
            self.assertEqual(len(tokenizer_calls), 1)
            self.assertTrue(any(keyword.arg == "add_special_tokens" and keyword.value.value is False
                                for keyword in tokenizer_calls[0].keywords))

    def test_sweep_resumes_completed_fits_and_evaluations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark_path = root / "atomic-v2.jsonl"
            write_jsonl(
                generate_atomic_benchmark(ATOMIC_SMOKE_V2_CONFIG), benchmark_path,
                seed=17, name=ATOMIC_SMOKE_V2_CONFIG.name,
            )
            source = cell_source("unique_pairs = {}")
            source = source.replace("import torch\n", "").replace("import torch.nn as nn\n", "")
            source = source.replace(
                "/content/amt/data/processed/amt_memorybench/amt-atomic-smoke-v2.jsonl",
                str(benchmark_path),
            )
            namespace = helper_namespace()
            with contextlib.redirect_stdout(io.StringIO()):
                exec(source, namespace)
            pairs = namespace["selected_pairs"]
            self.assertEqual(len(pairs), 12)
            self.assertEqual(len(namespace["unique_pairs"]), 32)
            for index in range(0, len(pairs), 2):
                self.assertNotEqual(pairs[index][0].answer, pairs[index + 1][0].answer)

            completed_fits = []
            generations = []
            interrupt = [True]
            answers = {namespace["memory_for"](train).text: train.answer for train, _ in pairs}

            def train(example, token_count, seed):
                if len(completed_fits) == 1 and interrupt[0]:
                    interrupt[0] = False
                    raise RuntimeError("simulated interruption")
                completed_fits.append((example.example_id, token_count, seed))
                return {
                    "prefix": example.answer, "losses": [1.0, 0.01], "trace": [],
                    "final_loss": 0.001, "answer_token_ids": [1, 2],
                    "answer_token_losses": [0.001], "eos_loss": 0.001,
                }

            def generate(prefix, question, memory_text=None):
                generations.append((prefix, question, memory_text))
                return answers[memory_text] if memory_text is not None else prefix or "unknown"

            fake_torch = SimpleNamespace(
                version=SimpleNamespace(cuda="test-cuda"),
                cuda=SimpleNamespace(
                    get_device_name=lambda index: "test-gpu",
                    reset_peak_memory_stats=lambda: None,
                    max_memory_allocated=lambda: 0,
                ),
                save=lambda value, path: path.write_bytes(pickle.dumps(value)),
                load=lambda path, **options: pickle.loads(path.read_bytes()),
                zeros_like=lambda value: "zero-control",
            )
            namespace.update({
                "torch": fake_torch, "hashlib": hashlib,
                "subprocess": SimpleNamespace(check_output=lambda args, **options: "test-commit" if args[-1] == "HEAD" and "rev-parse" in args else ""),
                "model": SimpleNamespace(config=SimpleNamespace(_commit_hash="test-model")),
                "model_id": "test-model", "MODEL_REVISION": "test-model",
                "PROMPT_STYLE": "qwen3-chat-nonthinking-short-span", "ANSWER_STOP_ID": 99,
                "quantization_config": SimpleNamespace(to_dict=lambda: {}), "dtype": "test-dtype",
                "TOKEN_COUNTS": (1, 2, 4), "SEEDS": (17, 23, 42), "TRAIN_STEPS": 100,
                "LEARNING_RATE": 0.01, "MAX_NEW_TOKENS": 16, "MAX_SESSION_HOURS": 6,
                "REQUIRE_TEXT_GATE": True, "TEXT_GATE_ALIAS_ACCURACY": 0.95,
                "RUN_ROOT": root / "runs", "platform": SimpleNamespace(python_version=lambda: "test-python"),
                "importlib": SimpleNamespace(metadata=SimpleNamespace(version=lambda name: "test-version")),
                "code_fingerprint": lambda: "test-helper", "train_one_free_token": train,
                "generate_from_prefix": generate,
            })
            sweep = cell_source("baseline_path =")
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    exec(sweep, namespace)
                exec(sweep, namespace)
                before = (len(completed_fits), len(generations))
                exec(sweep, namespace)
            self.assertEqual(len(completed_fits), 108)
            self.assertEqual(len(set(completed_fits)), 108)
            self.assertEqual(len(namespace["results"]), 108)
            self.assertEqual(len(namespace["baselines"]), 48)
            self.assertEqual(before, (len(completed_fits), len(generations)))
            self.assertTrue(all(item["test_correct"] for item in namespace["results"]))
            self.assertTrue(all(not item["scores"]["test_wrong"]["strict"] for item in namespace["results"]))


if __name__ == "__main__":
    unittest.main()