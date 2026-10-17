# SPDX-License-Identifier: MIT
# See LICENSE, LICENSE.fla and THIRD_PARTY_NOTICES.txt.
"""Load safetensors weights into Jittor without importing Torch."""

import json
import os
import struct
from dataclasses import asdict
from pathlib import Path

import jittor as jt
import numpy as np

from .models.cyfa import CyclicFlowAttentionConfig, CyclicFlowAttentionForCausalLM

_DTYPES = {
    "F32": "<f4",
    "F16": "<f2",
    "I64": "<i8",
    "I32": "<i4",
    "U8": "u1",
    "BOOL": "?",
}


def load_safetensors(path):
    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))
        offset = 8 + size
    result = {}
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        begin, end = meta["data_offsets"]
        dtype = meta["dtype"]
        raw = np.memmap(
            path,
            mode="r",
            dtype="<u2" if dtype == "BF16" else _DTYPES[dtype],
            offset=offset + begin,
            shape=tuple(meta["shape"]),
        )
        if dtype == "BF16":
            raw = (np.asarray(raw).astype(np.uint32) << 16).view(np.float32)
        result[key] = np.array(raw, copy=True)
    return result


def load_checkpoint(model, directory, strict=True):
    directory = Path(directory)
    weights = {}
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError("No safetensors weights under " + str(directory))
    for path in files:
        weights.update(load_safetensors(path))
    params = model.state_dict()
    missing = []
    for name, var in params.items():
        if name not in weights:
            missing.append(name)
            continue
        arr = weights.pop(name)
        if tuple(arr.shape) != tuple(var.shape):
            raise ValueError((name, arr.shape, var.shape))
        var.assign(jt.array(arr).cast(var.dtype))
    if strict and (missing or weights):
        raise ValueError({"missing": missing, "unexpected": list(weights)})
    return {"missing": missing, "unexpected": list(weights)}


def resolve_checkpoint(path, weights=True):
    local = Path(path).expanduser()
    if local.exists():
        return local
    from huggingface_hub import snapshot_download

    patterns = [
        "config.json",
        "*.safetensors",
        "*.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]
    if not weights:
        patterns = [p for p in patterns if "safetensors" not in p]
    return Path(snapshot_download(path, allow_patterns=patterns))


def read_config(path):
    path = Path(path)
    data = json.loads((path / "config.json" if path.is_dir() else path).read_text())
    if data.get("checkpoint_level", 0) != 0:
        raise ValueError("Only checkpoint_level=0 is supported")
    config = CyclicFlowAttentionConfig(
        **{
            k: v
            for k, v in data.items()
            if k in CyclicFlowAttentionConfig.__dataclass_fields__
        }
    )
    if (
        config.hidden_size <= 0
        or config.num_hidden_layers <= 0
        or config.vocab_size <= 1
    ):
        raise ValueError("Invalid model dimensions")
    if data.get("hidden_act", "swish") not in ("swish", "silu"):
        raise ValueError("Only the released SwiGLU model architecture is supported")
    return config


def load_model(path):
    directory = resolve_checkpoint(path)
    model = CyclicFlowAttentionForCausalLM(read_config(directory)).to_bfloat16()
    load_checkpoint(model, directory)
    return model, directory


def initialize_model(config, std=0.02):
    model = CyclicFlowAttentionForCausalLM(config)
    for name, value in model.state_dict().items():
        if name.endswith(".weight") and "norm" not in name:
            value.assign(jt.randn(value.shape) * std)
        elif name.endswith(".bias"):
            value.assign(jt.zeros_like(value))
    return model.to_bfloat16()


def save_safetensors(tensors, path):
    """Stream parameters one at a time, retaining each tensor's storage dtype."""
    path = Path(path)
    header, offset = {}, 0
    for name, value in tensors.items():
        dtype = str(value.dtype)
        code = {"bfloat16": "BF16", "float32": "F32", "float16": "F16"}[dtype]
        nbytes = int(np.prod(value.shape)) * (4 if code == "F32" else 2)
        header[name] = {
            "dtype": code,
            "shape": list(value.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        for name, value in tensors.items():
            code = header[name]["dtype"]
            array = value.float32().numpy()
            if code == "BF16":
                array = (array.view(np.uint32) >> 16).astype("<u2")
            else:
                array = array.astype("<f4" if code == "F32" else "<f2")
            stream.write(array.tobytes())
    os.replace(temporary, path)


def save_model(model, directory, tokenizer=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = asdict(model.config)
    config.update(
        model_type="cyclic_flow_attention",
        architectures=["CyclicFlowAttentionForCausalLM"],
    )
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if tokenizer is not None:
        config.update(eos_token_id=tokenizer.eos_id, bos_token_id=tokenizer.bos_id)
        (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    save_safetensors(model.state_dict(), directory / "model.safetensors")
    if tokenizer is not None:
        tokenizer.save(directory)


class Tokenizer:
    """Fast tokenizer.json backend, with a byte tokenizer for offline examples."""

    def __init__(self, path):
        self.byte = str(path) == "byte"
        self.backend = None
        self.eos_id, self.bos_id = 2, 1
        if not self.byte:
            from tokenizers import Tokenizer as FastTokenizer

            directory = resolve_checkpoint(path, weights=False)
            self.backend = FastTokenizer.from_file(str(directory / "tokenizer.json"))
            config = directory / "config.json"
            if config.exists():
                data = json.loads(config.read_text())
                self.eos_id, self.bos_id = (
                    data.get("eos_token_id", 2),
                    data.get("bos_token_id", 1),
                )
            special = directory / "tokenizer_config.json"
            if special.exists():
                data = json.loads(special.read_text())
                for name in ("eos", "bos"):
                    token = data.get(name + "_token")
                    if isinstance(token, dict):
                        token = token.get("content")
                    index = self.backend.token_to_id(token) if token else None
                    if index is not None:
                        setattr(self, name + "_id", index)
        if self.eos_id is None:
            raise ValueError("Tokenizer needs an EOS token")
        if self.bos_id is None:
            self.bos_id = self.eos_id

    def encode(self, text, add_special_tokens=False):
        return (
            list(text.encode("utf-8"))
            if self.byte
            else self.backend.encode(text, add_special_tokens=add_special_tokens).ids
        )

    def decode(self, ids):
        return (
            bytes(int(i) for i in ids).decode("utf-8", errors="replace")
            if self.byte
            else self.backend.decode(list(map(int, ids)), skip_special_tokens=True)
        )

    def save(self, directory):
        directory = Path(directory)
        if self.byte:
            (directory / "tokenizer_config.json").write_text(
                json.dumps({"cyfa_byte_tokenizer": True})
            )
        else:
            self.backend.save(str(directory / "tokenizer.json"))
            special = {
                name + "_token": self.backend.id_to_token(getattr(self, name + "_id"))
                for name in ("eos", "bos")
            }
            (directory / "tokenizer_config.json").write_text(json.dumps(special))


def load_tokenizer(path, override=None):
    if override:
        return Tokenizer(override)
    if str(path) == "byte":
        return Tokenizer("byte")
    directory = resolve_checkpoint(path, weights=False)
    config = directory / "tokenizer_config.json"
    if config.exists() and json.loads(config.read_text()).get("cyfa_byte_tokenizer"):
        return Tokenizer("byte")
    return Tokenizer(directory)


def setup_cuda(seed=42):
    jt.flags.use_cuda = 1
    jt.flags.use_tensorcore = 0
    jt.set_global_seed(seed)
