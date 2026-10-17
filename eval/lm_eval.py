# SPDX-License-Identifier: MIT
"""Use official lm-eval task definitions with native Jittor model execution."""

import argparse
import json
from pathlib import Path

from cyclic_flow_attention.generation import generate
from cyclic_flow_attention.utils import load_model, load_tokenizer, setup_cuda

from .scoring import Scorer


def make_adapter(scorer):
    from lm_eval.api.model import LM

    class JittorLM(LM):
        def __init__(self):
            super().__init__()

        @property
        def tokenizer_name(self):
            return "cyfa-jittor"

        def loglikelihood(self, requests, disable_tqdm=False):
            return scorer.loglikelihood([r.args for r in requests])

        def loglikelihood_rolling(self, requests, disable_tqdm=False):
            return scorer.rolling_loglikelihood([r.args[0] for r in requests])

        def generate_until(self, requests, disable_tqdm=False):
            results = []
            for request in requests:
                context, options = request.args
                until = options.get("until", [])
                until = [until] if isinstance(until, str) else until
                ids = scorer.tokenizer.encode(context)[-scorer.max_length :] or [
                    scorer.tokenizer.bos_id
                ]
                stop = lambda values: any(
                    s in scorer.tokenizer.decode(values) for s in until if s
                )
                values = generate(
                    scorer.model,
                    ids,
                    options.get("max_gen_toks", 48),
                    scorer.tokenizer.eos_id,
                    on_token=stop,
                )
                text = scorer.tokenizer.decode(values)
                for marker in until:
                    if marker:
                        text = text.split(marker)[0]
                results.append(text)
            return results

    return JittorLM()


def prepare_model(model):
    """Match the original HF loader's BF16 parameters and FP32 attention state."""
    for name, parameter in model.state_dict().items():
        if "norm.weight" in name and not any(
            key in name for key in ("q_norm.weight", "k_norm.weight")
        ):
            parameter.assign(parameter.bfloat16())
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()
    setup_cuda()
    model, directory = load_model(args.model_path)
    prepare_model(model)
    scorer = Scorer(
        model,
        load_tokenizer(directory, args.tokenizer),
        args.max_length,
        args.batch_size,
    )
    from lm_eval import simple_evaluate

    result = simple_evaluate(
        model=make_adapter(scorer),
        tasks=args.tasks.split(","),
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        log_samples=True,
    )
    path = Path(args.output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            result,
            indent=2,
            default=lambda value: (
                value.tolist() if hasattr(value, "tolist") else str(value)
            ),
        )
    )
    print(json.dumps(result["results"], indent=2))


if __name__ == "__main__":
    main()
