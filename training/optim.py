# SPDX-License-Identifier: MIT
# See LICENSE, LICENSE.fla and THIRD_PARTY_NOTICES.txt.
"""Native Jittor AdamW with FP32 master weights and stable parameter dtypes."""

import jittor as jt


class AdamW(jt.optim.Optimizer):
    """Keep moments and master weights in FP32; round model weights once per step."""

    def __init__(
        self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    ):
        if lr < 0 or eps <= 0 or weight_decay < 0 or not all(0 <= b < 1 for b in betas):
            raise ValueError("Invalid AdamW hyperparameters")
        super().__init__(list(params), lr)
        self.betas, self.eps, self.weight_decay = betas, eps, weight_decay
        self.update_step = 0
        for group in self.param_groups:
            self._initialize(group)
        self._have_grad = False

    def _initialize(self, group):
        group["master"] = [p.float32().clone().stop_grad() for p in group["params"]]
        group["m"] = [jt.zeros(p.shape, "float32").stop_grad() for p in group["params"]]
        group["v"] = [jt.zeros(p.shape, "float32").stop_grad() for p in group["params"]]
        group["parameter_dtypes"] = [str(p.dtype) for p in group["params"]]
        group["grads"] = [
            jt.zeros(p.shape, "float32").stop_grad() for p in group["params"]
        ]

    def add_param_group(self, group):
        self._initialize(group)
        super().add_param_group(group)

    def zero_grad(self):
        self._have_grad = False

    def backward(self, loss, retain_graph=False):
        parameters = [
            p
            for group in self.param_groups
            for p in group["params"]
            if not p.is_stop_grad()
        ]
        if not parameters:
            return
        jt.sync(parameters)
        gradients = [
            g.float32().stop_grad() for g in jt.grad(loss, parameters, retain_graph)
        ]
        if jt.in_mpi:
            # Jittor 1.3.11 has no BF16 NCCL all-reduce dispatch. One FP32
            # collective also avoids scheduling hundreds of small reductions.
            reduced = jt.concat([g.reshape((-1,)) for g in gradients]).mpi_all_reduce(
                "mean"
            )
            offset = 0
            for i, parameter in enumerate(parameters):
                size = parameter.numel()
                gradients[i] = reduced[offset : offset + size].reshape(parameter.shape)
                offset += size
        cursor = 0
        for group in self.param_groups:
            for parameter, buffer in zip(group["params"], group["grads"]):
                if parameter.is_stop_grad():
                    continue
                gradient = gradients[cursor]
                buffer.update(buffer + gradient if self._have_grad else gradient)
                cursor += 1
        self._have_grad = True
        self.n_step += 1

    def clip_grad_norm(self, max_norm, norm_type=2):
        if norm_type != 2 or max_norm <= 0:
            raise ValueError("Use a positive maximum L2 gradient norm")
        if not self._have_grad:
            return
        gradients = [
            g
            for group in self.param_groups
            for p, g in zip(group["params"], group["grads"])
            if not p.is_stop_grad()
        ]
        norm = jt.sqrt(sum((g * g).sum() for g in gradients))
        coefficient = jt.minimum(max_norm / (norm + 1e-6), 1.0)
        for gradient in gradients:
            gradient.update(gradient * coefficient)

    def step(self, loss=None, retain_graph=False):
        self.pre_step(loss, retain_graph)
        try:
            self.update_step += 1
            for group in self.param_groups:
                lr = group.get("lr", self.lr)
                b0, b1 = group.get("betas", self.betas)
                eps = group.get("eps", self.eps)
                decay = group.get("weight_decay", self.weight_decay)
                bc1 = 1 - b0**self.update_step
                bc2 = 1 - b1**self.update_step
                for p, g, master, m, v, dtype in zip(
                    group["params"],
                    group["grads"],
                    group["master"],
                    group["m"],
                    group["v"],
                    group["parameter_dtypes"],
                ):
                    if p.is_stop_grad():
                        continue
                    g = g.float32()
                    m.update((b0 * m + (1 - b0) * g).float32())
                    v.update((b1 * v + (1 - b1) * g * g).float32())
                    master.update(
                        (
                            master * (1 - lr * decay)
                            - (lr / bc1) * m / (jt.sqrt(v / bc2) + eps)
                        ).float32()
                    )
                    p.update(master.cast(dtype))
                    # FP32 cast can alias the stopped master; keep model leaves trainable.
                    p.start_grad()
        finally:
            self.post_step()
