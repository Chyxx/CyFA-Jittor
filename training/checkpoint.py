# SPDX-License-Identifier: MIT
"""Resumable training checkpoints, using safetensors and JSON only."""

import json
import os
from pathlib import Path

import jittor as jt

from cyclic_flow_attention.utils import load_safetensors, save_model, save_safetensors


def save_training(model, optimizer, tokenizer, directory, step, rng, arguments):
    directory = Path(directory)
    temporary = directory.with_name(directory.name + ".incomplete")
    if directory.exists() or temporary.exists():
        raise FileExistsError(str(directory))
    temporary.mkdir(parents=True)
    save_model(model, temporary, tokenizer)
    tensors = {}
    names = [[p.name() for p in group["params"]] for group in optimizer.param_groups]
    for g, group in enumerate(optimizer.param_groups):
        for kind in ("master", "m", "v"):
            tensors.update(
                {f"{g}.{kind}.{i}": value for i, value in enumerate(group[kind])}
            )
    (temporary / "training_state").mkdir()
    save_safetensors(tensors, temporary / "training_state" / "optimizer.safetensors")
    state = dict(
        step=step,
        update_step=optimizer.update_step,
        rng=rng.bit_generator.state,
        arguments=arguments,
        parameter_names=names,
        world_size=jt.world_size if jt.in_mpi else 1,
    )
    (temporary / "training.json").write_text(json.dumps(state, indent=2) + "\n")
    os.replace(temporary, directory)


def restore_training(optimizer, directory, rng):
    directory = Path(directory)
    state = json.loads((directory / "training.json").read_text())
    world_size = jt.world_size if jt.in_mpi else 1
    if state["world_size"] != world_size:
        raise ValueError("Resume requires the same number of data-parallel ranks")
    names = [[p.name() for p in group["params"]] for group in optimizer.param_groups]
    if state["parameter_names"] != names:
        raise ValueError("Optimizer parameter names/order differ")
    tensors = load_safetensors(directory / "training_state" / "optimizer.safetensors")
    for g, group in enumerate(optimizer.param_groups):
        for kind in ("master", "m", "v"):
            for i, value in enumerate(group[kind]):
                array = tensors.pop(f"{g}.{kind}.{i}")
                if tuple(array.shape) != tuple(value.shape):
                    raise ValueError("Optimizer parameter shape mismatch")
                value.assign(jt.array(array).stop_grad())
    if tensors:
        raise ValueError("Unexpected optimizer tensors")
    optimizer.update_step = state["update_step"]
    optimizer.n_step = state["step"] * state["arguments"]["gradient_accumulation_steps"]
    rng.bit_generator.state = state["rng"]
    return state
