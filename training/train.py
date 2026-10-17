# SPDX-License-Identifier: MIT
"""Train or resume a native Jittor causal language model."""

import argparse
import json
import math
import time
from pathlib import Path

import jittor as jt
import numpy as np

from cyclic_flow_attention.utils import (
    initialize_model,
    load_model,
    load_tokenizer,
    read_config,
    setup_cuda,
)
from training.optim import AdamW

from .checkpoint import restore_training, save_training
from .data import TokenDataset


def learning_rate(step, total, warmup, peak, minimum):
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup - 1, 1)
    return peak * (
        minimum + (1 - minimum) * 0.5 * (1 + math.cos(math.pi * min(progress, 1)))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config")
    source.add_argument("--model-path")
    source.add_argument("--resume")
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=30720)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=1024)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-steps", type=int, default=1024)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (
        min(
            args.context_length,
            args.batch_size,
            args.gradient_accumulation_steps,
            args.max_steps,
            args.save_steps,
            args.logging_steps,
        )
        < 1
    ):
        parser.error("Lengths, batches, steps and intervals must be positive")
    setup_cuda(args.seed)
    from jittor import nn

    rank = jt.rank if jt.in_mpi else 0
    world = jt.world_size if jt.in_mpi else 1
    rng = np.random.default_rng(args.seed)
    dataset = TokenDataset(args.data)
    recipe = dict(vars(args), dataset_metadata=dataset.metadata)
    if args.config:
        model = initialize_model(read_config(args.config))
        tokenizer = load_tokenizer(args.tokenizer or dataset.metadata["tokenizer"])
    else:
        model, directory = load_model(args.resume or args.model_path)
        tokenizer = load_tokenizer(directory, args.tokenizer)
    if jt.in_mpi:
        for parameter in model.parameters():
            parameter.assign(parameter.mpi_broadcast())
            parameter.start_grad()
        jt.sync_all(True)
    optimizer = AdamW(
        model.parameters(),
        args.learning_rate,
        (args.beta1, args.beta2),
        args.epsilon,
        args.weight_decay,
    )
    start = 0
    if args.resume:
        state = restore_training(optimizer, args.resume, rng)
        start = state["step"]
        keys = (
            "context_length",
            "batch_size",
            "gradient_accumulation_steps",
            "max_steps",
            "learning_rate",
            "min_lr_ratio",
            "warmup_steps",
            "weight_decay",
            "beta1",
            "beta2",
            "epsilon",
            "seed",
            "max_grad_norm",
        )
        changed = [k for k in keys if state["arguments"][k] != vars(args)[k]]
        if state["arguments"].get("dataset_metadata") != dataset.metadata:
            parser.error("Resume dataset metadata differs")
        if changed:
            parser.error("Resume recipe differs: " + ", ".join(changed))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(output.glob("checkpoint-*")):
        parser.error("Output already has checkpoints; resume or choose a new directory")
    model.train()
    for step in range(start, args.max_steps):
        tick = time.perf_counter()
        optimizer.lr = learning_rate(
            step,
            args.max_steps,
            args.warmup_steps,
            args.learning_rate,
            args.min_lr_ratio,
        )
        loss_value = 0.0
        for _ in range(args.gradient_accumulation_steps):
            # All ranks draw the same global batch, then take disjoint local slices.
            batch = dataset.batch(rng, args.batch_size * world, args.context_length)
            batch = batch[rank * args.batch_size : (rank + 1) * args.batch_size]
            if int(batch.max()) >= model.config.vocab_size:
                raise ValueError("Dataset tokenizer exceeds model vocabulary")
            ids = jt.array(batch)
            logits = model(ids[:, :-1])["logits"].float32()
            loss = nn.cross_entropy_loss(
                logits.reshape((-1, model.config.vocab_size)), ids[:, 1:].reshape((-1,))
            )
            value = float(loss.item())
            if not math.isfinite(value):
                raise FloatingPointError(f"Nonfinite loss at step {step + 1}")
            loss_value += value / args.gradient_accumulation_steps
            optimizer.backward(loss / args.gradient_accumulation_steps)
        if args.max_grad_norm > 0:
            optimizer.clip_grad_norm(args.max_grad_norm)
        optimizer.step()
        jt.sync_all(True)
        if jt.in_mpi and (step + 1) % args.logging_steps == 0:
            loss_value = float(jt.array([loss_value]).mpi_all_reduce("mean").item())
        if rank == 0 and (step + 1) % args.logging_steps == 0:
            record = dict(
                step=step + 1,
                loss=loss_value,
                learning_rate=optimizer.lr,
                tokens=(step + 1)
                * args.context_length
                * args.batch_size
                * world
                * args.gradient_accumulation_steps,
                seconds=time.perf_counter() - tick,
            )
            print(json.dumps(record), flush=True)
            with (output / "train.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
        if (step + 1) % args.save_steps == 0 or step + 1 == args.max_steps:
            if rank == 0:
                save_training(
                    model,
                    optimizer,
                    tokenizer,
                    output / f"checkpoint-{step + 1}",
                    step + 1,
                    rng,
                    recipe,
                )
            if jt.in_mpi:
                jt.sync_all(True)


if __name__ == "__main__":
    main()
