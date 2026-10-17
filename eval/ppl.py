# SPDX-License-Identifier: MIT
"""Token-weighted perplexity over nonoverlapping pretokenized sequences."""

import argparse
import json
import math
from pathlib import Path

from cyclic_flow_attention.utils import load_model, load_tokenizer, setup_cuda
from training.data import TokenDataset

from .scoring import Scorer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-path")
    args = parser.parse_args()
    setup_cuda()
    model, directory = load_model(args.model_path)
    scorer = Scorer(
        model,
        load_tokenizer(directory, args.tokenizer),
        args.context_length,
        args.batch_size,
    )
    tokens = TokenDataset(args.data).tokens
    total, count, pending = 0.0, 0, []
    for index, start in enumerate(range(1, len(tokens), args.context_length)):
        if args.limit is not None and index >= args.limit:
            break
        target = list(map(int, tokens[start : start + args.context_length]))
        pending.append(([int(tokens[start - 1])], target))
        if len(pending) == args.batch_size:
            total += sum(x[0] for x in scorer.token_loglikelihood(pending))
            count += sum(len(x[1]) for x in pending)
            pending = []
    if pending:
        total += sum(x[0] for x in scorer.token_loglikelihood(pending))
        count += sum(len(x[1]) for x in pending)
    if not count:
        parser.error("No evaluation tokens selected")
    result = dict(
        tokens=count,
        negative_log_likelihood=-total / count,
        perplexity=math.exp(min(-total / count, 709)),
        context_length=args.context_length,
    )
    if args.output_path:
        path = Path(args.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
