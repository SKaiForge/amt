# AMT Memory Lab

Infrastructure for Atomic Memory Token experiments, beginning with the cleaned
LongMemEval benchmark.

## LongMemEval

The source is the official cleaned release:
`xiaowu0162/longmemeval-cleaned`.

### Benchmark roles

- **LongMemEval-S (primary):** The main end-to-end memory benchmark and primary
	number for comparison with existing memory architectures.
- **LongMemEval-Oracle (diagnostic):** Isolates latent-memory representation and
	reasoning quality from retrieval quality. Use this first while developing the
	postulate extractor and AMT representation.
- **LongMemEval-M (scaling):** Tests whether latent memory remains efficient as
	history grows to roughly 1.5 million tokens.

The experimental order is Oracle, then S, then M. The reporting priority is S,
with Oracle as a diagnostic and M as the scaling result.

```bash
PYTHONPATH=src python3 -m datasets.longmemeval download oracle
PYTHONPATH=src python3 -m datasets.longmemeval inspect oracle --limit 5
```

Each download command fetches one variant. Fetch the larger variants separately,
or request all three sequentially:

```bash
PYTHONPATH=src python3 -m datasets.longmemeval download s
PYTHONPATH=src python3 -m datasets.longmemeval download m
PYTHONPATH=src python3 -m datasets.longmemeval download all
```

The complete release is approximately 3 GB, so `all` is intentionally opt-in.

Raw datasets are stored under `data/raw/longmemeval/` and excluded from Git.
Each download also writes a manifest containing its source URL, timestamp, byte
size, and SHA-256 digest.

The loader validates each record while streaming the top-level JSON array, so
the larger `s` and `m` variants do not need to be held entirely in memory.

## Postulate extraction pilot

Generate a deterministic Oracle pilot with eight non-abstention records from
each LongMemEval question type:

```bash
PYTHONPATH=src python3 -m datasets.longmemeval extract-pilot \
	--per-type 8 \
	--seed 17
```

The output is written to
`data/processed/longmemeval/oracle_postulates.jsonl`. Oracle labels select the
evidence turns, but the extractor receives only session IDs, dates, roles, turn
indexes, and message content. Questions, answers, evidence flags, and answer
session IDs are not exposed to it.

The current `sentence-candidate-v1` extractor is an infrastructure baseline. It
splits evidence turns into low-confidence (`0.25`) fact candidates with a
generic `states` predicate. Its output is suitable for testing provenance,
storage, and review workflows, but it is not yet the final semantic postulate
extractor.

Generate the separate, fully offline deterministic semantic artifact with:

```bash
PYTHONPATH=src python3 -m datasets.longmemeval extract-pilot \
	--extractor offline-semantic \
	--per-type 8 \
	--seed 17
```

This writes `data/processed/longmemeval/oracle_postulates_semantic.jsonl` and
does not change the sentence-candidate file. The `offline-semantic-v1` extractor
inspects every sanitized user and assistant turn, retains first-person durable
user claims, and retains assistant content only for explicit recommendations,
commitments, or recalled information. It deterministically classifies retained
claims as facts, preferences, goals, constraints, events, or relationships and
records the assertion role and exact source turn.

This rule-based artifact is a reproducible semantic baseline. Conservative
rules may emit no memory for a conversation that contains no recognized durable
claim; they intentionally favor precision over forcing coverage.

## AMT-MemoryBench

AMT-MemoryBench is a controlled synthetic benchmark for testing latent memory
access independently of conversational postulate extraction. It uses gold
postulates and records relevant and required memory IDs for single-memory,
multi-memory, random-distractor, semantic-distractor, and conditional-
preference tasks.

Generate the small smoke tier (10 users, 20 memories per user, 100 examples):

```bash
PYTHONPATH=src python3 -m datasets.amt_memorybench smoke
```

Generate AMT-100 (100 users, 50 memories per user, 10,000 examples):

```bash
PYTHONPATH=src python3 -m datasets.amt_memorybench amt-100
```

Both commands use seed 17 by default and write JSONL artifacts under
`data/processed/amt_memorybench/`. Use `--seed` or `--output` to produce a
reproducible alternate run or location.

Evaluate the current transparent retrieval baselines against an existing tier:

```bash
PYTHONPATH=src python3 -m datasets.amt_memorybench smoke --evaluate --selector none
PYTHONPATH=src python3 -m datasets.amt_memorybench smoke --evaluate --selector random
PYTHONPATH=src python3 -m datasets.amt_memorybench smoke --evaluate --selector lexical
```

Use `--k` to change the number of selected memories. The `none` selector is a
no-memory control, `random` is a deterministic chance baseline, and `lexical`
selects memories by query/text token overlap.

The text-memory experiment can use `TextMemoryAnswerer` from
`datasets.amt_memorybench.answer`. Provide any model adapter with a
`generate(prompt: str) -> str` method; the harness formats selected memories,
builds the prompt, and reports exact-answer accuracy through `evaluate_answers`.

### AMT H1/H3 experiment

The first AMT experiment bypasses retrieval and uses each example's required
gold postulate. `amt_memory.models.QwenPrefixAMT` freezes a Qwen reader, pools
each postulate through its frozen input embeddings, projects it to one continuous
token, and prepends those tokens to the query embeddings. This tests one-token
atomic memory capacity (H1) and direct latent-token consumption by the decoder
(H3); it intentionally does not yet add cross-attention or CPT alignment.

On a GPU runtime, use the `single_recall` AMT-MemoryBench examples first. For
each example, select `required_memory_ids` directly, train only
`QwenPrefixAMT.parameters()` with `amt.loss([postulate.text], prompt, answer)`,
and evaluate with `amt.generate`. A successful smoke result requires both a
falling training loss and answers that contain the gold value; cross-attention,
retrieval, and multi-memory composition are separate later experiments.