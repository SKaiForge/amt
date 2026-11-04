"""CLI for generating controlled AMT benchmark artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from datasets.amt_memorybench.generate import (
    ATOMIC100_CONFIG,
    ATOMIC_SMOKE_CONFIG,
    ATOMIC_SMOKE_V2_CONFIG,
    AtomicBenchmarkConfig,
    generate_atomic_benchmark,
    AMT100_CONFIG,
    SMOKE_CONFIG,
    generate_benchmark,
    write_jsonl,
)
from datasets.amt_memorybench.evaluate import (
    LexicalMemorySelector,
    NoMemorySelector,
    RandomMemorySelector,
    evaluate_selection,
    read_jsonl,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="amt-memorybench")
    parser.add_argument(
        "tier",
        choices=(
            "smoke",
            "amt-100",
            "atomic-smoke",
            "atomic-smoke-v2",
            "atomic-100",
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument(
        "--selector", choices=("lexical", "random", "none"), default="lexical"
    )
    args = parser.parse_args(argv)

    base = {
        "smoke": SMOKE_CONFIG,
        "amt-100": AMT100_CONFIG,
        "atomic-smoke": ATOMIC_SMOKE_CONFIG,
        "atomic-smoke-v2": ATOMIC_SMOKE_V2_CONFIG,
        "atomic-100": ATOMIC100_CONFIG,
    }[args.tier]
    if args.seed is None:
        config = base
    elif args.tier.startswith("atomic"):
        config = AtomicBenchmarkConfig(name=base.name, users=base.users, seed=args.seed)
    else:
        config = BenchmarkConfig(
            name=base.name,
            users=base.users,
            memories_per_user=base.memories_per_user,
            queries_per_task=base.queries_per_task,
            seed=args.seed,
        )
    output = args.output or Path(
        f"data/processed/amt_memorybench/{config.name}.jsonl"
    )
    if args.evaluate:
        selector = {
            "lexical": LexicalMemorySelector(),
            "random": RandomMemorySelector(seed=config.seed),
            "none": NoMemorySelector(),
        }[args.selector]
        metrics = evaluate_selection(
            read_jsonl(output), selector, k=args.k
        )
        print(json.dumps(asdict(metrics), indent=2))
        return 0
    examples = (
        generate_atomic_benchmark(config)
        if args.tier.startswith("atomic")
        else generate_benchmark(config)
    )
    count = write_jsonl(examples, output, seed=config.seed, name=config.name)
    print(json.dumps({"benchmark": config.name, "output": str(output), "examples": count}, indent=2))
    return 0