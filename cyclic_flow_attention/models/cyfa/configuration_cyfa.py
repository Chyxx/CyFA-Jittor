# SPDX-License-Identifier: MIT
# See LICENSE, LICENSE.fla and THIRD_PARTY_NOTICES.txt.
"""Configuration of the released CyFA language models."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class CyclicFlowAttentionConfig:
    hidden_size: int = 1024
    num_heads: int = 4
    head_dim: int = 256
    expand_k: float = 1.0
    expand_v: float = 1.0
    num_slots: int = 128
    num_hidden_layers: int = 24
    vocab_size: int = 32000
    hidden_ratio: float = 4.0
    intermediate_size: int | None = None
    use_short_conv: bool = True
    conv_size: int = 4
    conv_bias: bool = False
    norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    chunk_size: int = 64

    @classmethod
    def from_pretrained(cls, path):
        from ...utils import read_config, resolve_checkpoint

        local = Path(path)
        return read_config(
            local if local.exists() else resolve_checkpoint(path, weights=False)
        )
