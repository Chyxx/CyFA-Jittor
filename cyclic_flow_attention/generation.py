# SPDX-License-Identifier: MIT
# See LICENSE, LICENSE.fla and THIRD_PARTY_NOTICES.txt.
"""Autoregressive generation with native constant-size recurrent state."""

import argparse

import jittor as jt
import numpy as np

from .utils import load_model, load_tokenizer, setup_cuda


def _detach_cache(value):
    if isinstance(value, jt.Var):
        return value.detach().stop_grad()
    if isinstance(value, dict):
        return {k: _detach_cache(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_detach_cache(v) for v in value)
    return value


def generate(
    model,
    ids,
    max_new_tokens=64,
    eos_token_id=None,
    temperature=0.0,
    top_p=1.0,
    seed=42,
    on_token=None,
):
    if not ids or max_new_tokens < 0 or temperature < 0 or not 0 < top_p <= 1:
        raise ValueError("Invalid prompt or sampling options")
    model.eval()
    rng = np.random.default_rng(seed)
    current, cache, generated = list(ids), None, []
    with jt.no_grad():
        for _ in range(max_new_tokens):
            result = model(
                jt.array(np.array(current, np.int32)[None]), cache=cache, use_cache=True
            )
            logits = result["logits"][0, -1].float32().numpy().astype(np.float64)
            cache = _detach_cache(result["cache"])
            if temperature == 0:
                token = int(logits.argmax())
            else:
                logits /= temperature
                probs = np.exp(logits - logits.max())
                probs /= probs.sum()
                order = np.argsort(-probs)
                keep = np.cumsum(probs[order]) - probs[order] < top_p
                order = order[keep]
                weights = probs[order] / probs[order].sum()
                token = int(rng.choice(order, p=weights))
            generated.append(token)
            if on_token is not None and on_token(generated):
                break
            if token == eos_token_id:
                break
            current = [token]
    return generated


def generate_batch(model, prompts, max_new_tokens=48, eos_token_id=None, on_token=None):
    """Greedy generation for right-padded prompts with independent stopping."""
    if not prompts or any(not ids for ids in prompts) or max_new_tokens < 0:
        raise ValueError(
            "Expected nonempty prompts and a nonnegative generation length"
        )
    model.eval()
    lengths = np.array([len(ids) for ids in prompts], np.int32)
    width = int((lengths.max() + 63) // 64 * 64)
    ids = np.zeros((len(prompts), width), np.int32)
    mask = np.zeros_like(ids)
    for row, tokens in enumerate(prompts):
        ids[row, : len(tokens)] = tokens
        mask[row, : len(tokens)] = 1
    generated = [[] for _ in prompts]
    finished = np.zeros(len(prompts), bool)
    with jt.no_grad():
        hidden, cache = model.model(
            jt.array(ids), use_cache=True, attention_mask=jt.array(mask)
        )
        last = hidden[jt.arange(len(prompts)).int32(), jt.array(lengths - 1)]
        logits = model.lm_head(last)
        cache = _detach_cache(cache)
        for step in range(max_new_tokens):
            tokens = logits.argmax(-1)[0].int32().numpy()
            for row, token in enumerate(tokens):
                if finished[row]:
                    continue
                generated[row].append(int(token))
                finished[row] = token == eos_token_id or bool(
                    on_token and on_token(generated[row])
                )
            if finished.all() or step + 1 == max_new_tokens:
                break
            result = model(jt.array(tokens[:, None]), cache=cache, use_cache=True)
            cache = _detach_cache(result["cache"])
            logits = result["logits"][:, -1]
    return generated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    setup_cuda(args.seed)
    model, directory = load_model(args.model_path)
    tokenizer = load_tokenizer(directory, args.tokenizer)
    ids = tokenizer.encode(args.prompt, add_special_tokens=True) or [tokenizer.bos_id]
    result = generate(
        model,
        ids,
        args.max_new_tokens,
        tokenizer.eos_id,
        args.temperature,
        args.top_p,
        args.seed,
    )
    print(tokenizer.decode(ids + result))


if __name__ == "__main__":
    main()
