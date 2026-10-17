import json
import subprocess
import sys
from pathlib import Path

import jittor as jt
import numpy as np

from cyclic_flow_attention.utils import load_safetensors
from training.optim import AdamW

ROOT = Path(__file__).resolve().parents[1]


def test_master_adamw_remains_trainable():
    values = [np.array([1.0, -0.5], np.float32), np.array([0.25, -1.0], np.float32)]
    params = [jt.array(values[0]), jt.array(values[1]).bfloat16()]
    optimizer = AdamW(params, lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01)
    masters = [v.copy() for v in values]
    moments = [np.zeros_like(v) for v in values]
    variances = [np.zeros_like(v) for v in values]
    for step in range(1, 4):
        gradients = [2 * p.float32().numpy() for p in params]
        loss = sum((p.float32() * p.float32()).sum() for p in params)
        optimizer.step(loss)
        jt.sync_all(True)
        for i, (p, gradient) in enumerate(zip(params, gradients)):
            moments[i] = 0.9 * moments[i] + 0.1 * gradient
            variances[i] = 0.95 * variances[i] + 0.05 * gradient * gradient
            masters[i] = masters[i] * (1 - 1e-5) - (1e-3 / (1 - 0.9**step)) * moments[
                i
            ] / (np.sqrt(variances[i] / (1 - 0.95**step)) + 1e-8)
            assert not p.is_stop_grad()
            np.testing.assert_allclose(
                optimizer.param_groups[0]["master"][i].numpy(),
                masters[i],
                rtol=1e-6,
                atol=1e-7,
            )


def test_continuous_and_resumed_training(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    tokens = np.tile(np.arange(32, 96, dtype="<u4"), 16)
    tokens.tofile(data / "tokens.bin")
    (data / "metadata.json").write_text(
        json.dumps(dict(tokens=len(tokens), tokenizer="byte"))
    )

    def run(source, output):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "training.train",
                *source,
                "--data",
                str(data),
                "--output-dir",
                str(output),
                "--context-length",
                "64",
                "--batch-size",
                "1",
                "--gradient-accumulation-steps",
                "2",
                "--max-steps",
                "4",
                "--save-steps",
                "2",
                "--warmup-steps",
                "0",
            ],
            check=True,
            cwd=ROOT,
        )

    continuous, resumed = tmp_path / "continuous", tmp_path / "resumed"
    run(
        ["--config", str(ROOT / "configs/tiny.json"), "--tokenizer", "byte"], continuous
    )
    run(["--resume", str(continuous / "checkpoint-2")], resumed)
    for name in ("model.safetensors", "training_state/optimizer.safetensors"):
        left = load_safetensors(continuous / "checkpoint-4" / name)
        right = load_safetensors(resumed / "checkpoint-4" / name)
        error = sum(np.sum((left[k].astype(np.float64) - right[k]) ** 2) for k in left)
        norm = sum(np.sum(left[k].astype(np.float64) ** 2) for k in left)
        assert np.sqrt(error / norm) < 1e-3
    losses = [
        json.loads(line)["loss"]
        for line in (continuous / "train.jsonl").read_text().splitlines()
    ]
    assert losses[-1] < losses[0]
    state = json.loads((resumed / "checkpoint-4/training.json").read_text())
    assert state["step"] == state["update_step"] == 4
