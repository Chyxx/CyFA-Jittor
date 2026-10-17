# SPDX-License-Identifier: MIT
"""Stream documents to a portable flat uint32 token file."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from cyclic_flow_attention.utils import Tokenizer


def documents(args):
    if args.input:
        path = Path(args.input)
        with path.open() as stream:
            for line in stream:
                yield (
                    json.loads(line)[args.text_column]
                    if path.suffix == ".jsonl"
                    else line.rstrip("\n")
                )
    else:
        from datasets import load_dataset

        dataset = load_dataset(
            args.dataset, args.dataset_config, split=args.split, streaming=True
        )
        for row in dataset:
            yield row[args.text_column]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="UTF-8 text (one document per line) or JSONL")
    source.add_argument("--dataset", help="Hugging Face dataset identifier")
    parser.add_argument("--dataset-config")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-documents", type=int)
    args = parser.parse_args()
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "tokens.bin").exists():
        parser.error("tokens.bin already exists; choose another output directory")
    tokenizer = Tokenizer(args.tokenizer)
    count, total = 0, 0
    temporary = directory / "tokens.bin.tmp"
    with temporary.open("wb") as stream:
        for text in documents(args):
            if args.max_documents is not None and count >= args.max_documents:
                break
            ids = tokenizer.encode(text) + [tokenizer.eos_id]
            np.asarray(ids, dtype="<u4").tofile(stream)
            total += len(ids)
            count += 1
    if total < 2:
        temporary.unlink()
        parser.error("Dataset must contain at least two tokens")
    os.replace(temporary, directory / "tokens.bin")
    metadata = dict(
        dtype="uint32",
        byte_order="little",
        tokens=total,
        documents=count,
        tokenizer=args.tokenizer,
        eos_token_id=tokenizer.eos_id,
    )
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
