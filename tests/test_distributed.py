import jittor as jt
import numpy as np
import pytest

from training.optim import AdamW


@pytest.mark.skipif(not jt.in_mpi, reason="Run with mpirun and NCCL")
def test_fp32_gradient_collective():
    parameter = jt.array([1.0]).bfloat16()
    optimizer = AdamW([parameter], lr=1e-3)
    for factor in (1.0, 2.0):
        optimizer.backward((parameter.float32() * (jt.rank + 1) * factor).sum())
        gradient = optimizer.param_groups[0]["grads"][0]
        assert str(gradient.dtype) == "float32"
        np.testing.assert_allclose(
            gradient.numpy(), [(jt.world_size + 1) * factor / 2], atol=1e-6
        )
        optimizer.step()
        jt.sync_all(True)
        assert not parameter.is_stop_grad()
