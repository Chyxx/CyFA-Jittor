# SPDX-License-Identifier: MIT
import json
from pathlib import Path

import numpy as np


class TokenDataset:
    def __init__(self, directory):
        directory = Path(directory)
        self.metadata = json.loads((directory / "metadata.json").read_text())
        self.tokens = np.memmap(directory / "tokens.bin", mode="r", dtype="<u4")
        if len(self.tokens) != self.metadata["tokens"] or len(self.tokens) < 2:
            raise ValueError("Token file and metadata do not match")

    def batch(self, rng, batch_size, context_length):
        if len(self.tokens) <= context_length:
            raise ValueError("Dataset must contain more tokens than context_length")
        starts = rng.integers(0, len(self.tokens) - context_length, size=batch_size)
        return np.stack(
            [self.tokens[i : i + context_length + 1] for i in starts]
        ).astype(np.int32)
