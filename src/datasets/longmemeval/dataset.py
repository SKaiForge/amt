"""Download and load the cleaned LongMemEval release."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from datetime import datetime, timezone
from json import JSONDecodeError
from pathlib import Path
from typing import Any, TextIO
from urllib.request import Request, urlopen

from datasets.longmemeval.schema import LongMemEvalRecord

BASE_URL = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"
DATASET_FILES = {
    "oracle": "longmemeval_oracle.json",
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
}


def download(variant: str, data_dir: Path, *, force: bool = False) -> Path:
    """Download one official dataset file and write a local provenance manifest."""
    filename = _filename(variant)
    destination = data_dir / filename
    if destination.exists() and not force:
        return destination

    data_dir.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    url = f"{BASE_URL}/{filename}"
    request = Request(url, headers={"User-Agent": "amt-memory-lab/0.1"})
    digest = hashlib.sha256()
    size = 0

    try:
        with urlopen(request) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    manifest = {
        "variant": variant,
        "filename": filename,
        "source_url": url,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "size_bytes": size,
        "sha256": digest.hexdigest(),
    }
    manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return destination


def dataset_path(variant: str, data_dir: Path) -> Path:
    return data_dir / _filename(variant)


def iter_records(path: Path) -> Iterator[LongMemEvalRecord]:
    """Validate and yield records without loading the complete JSON array."""
    with path.open("r", encoding="utf-8") as source:
        for index, value in enumerate(_iter_json_array(source)):
            if not isinstance(value, dict):
                raise ValueError(f"record {index} must be a JSON object")
            try:
                yield LongMemEvalRecord.from_dict(value)
            except ValueError as error:
                raise ValueError(f"invalid record {index}: {error}") from error


def _iter_json_array(source: TextIO, chunk_size: int = 1024 * 1024) -> Iterator[Any]:
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    started = False
    finished = False

    while not finished:
        chunk = source.read(chunk_size)
        at_eof = chunk == ""
        buffer = buffer[position:] + chunk
        position = 0

        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1

            if not started:
                if position >= len(buffer):
                    break
                if buffer[position] != "[":
                    raise ValueError("dataset root must be a JSON array")
                started = True
                position += 1
                continue

            while position < len(buffer) and (
                buffer[position].isspace() or buffer[position] == ","
            ):
                position += 1
            if position >= len(buffer):
                break
            if buffer[position] == "]":
                position += 1
                finished = True
                break

            try:
                value, position = decoder.raw_decode(buffer, position)
            except JSONDecodeError:
                if at_eof:
                    raise ValueError("dataset contains incomplete JSON") from None
                break
            yield value

        if at_eof:
            if not finished:
                raise ValueError("dataset JSON array is not closed")
            if buffer[position:].strip():
                raise ValueError("unexpected content after dataset JSON array")


def _filename(variant: str) -> str:
    try:
        return DATASET_FILES[variant]
    except KeyError:
        choices = ", ".join(DATASET_FILES)
        raise ValueError(f"unknown variant {variant!r}; choose one of: {choices}") from None