# SPDX-License-Identifier: MIT
# See LICENSE, LICENSE.fla and THIRD_PARTY_NOTICES.txt.
"""Chunkwise CyFA and scalar-attention autograd functions."""

import math
import os
from functools import lru_cache

import jittor as jt

from .chunk_bwd import scalar_grad
from .chunk_h import state_forward
from .chunk_o import output_forward
from .chunk_readout import first_readout
from .qk_norm import head_rmsnorm
from .utils import (
    affine_scan,
    bmm,
    clock_cumsum,
    clock_total,
    head_gates,
    head_layout,
    local_log2_cumsum,
    modal_readout,
    phase_write,
    use_large,
)


class ScalarAttention(jt.Function):
    def execute(self, q, k, v, g, initial, scale, raw_output, output_bth, v_bth):
        self.v_bth = v_bth
        self.output_bth = output_bth
        self.inputs = q, k, v, g, initial
        self.scale = scale
        self.raw_output = raw_output
        self.gc = local_log2_cumsum(g)
        self.states, final = state_forward(k, v, self.gc, initial, v_bth=v_bth)
        return output_forward(
            q,
            k,
            v,
            self.gc,
            self.states,
            scale,
            raw_output,
            output_bth=output_bth,
            v_bth=v_bth,
        ), final

    def grad(self, dy, df):
        from .utils import head_layout

        if self.output_bth and dy is not None:
            dy = head_layout(dy.bfloat16())
        saved = self.inputs
        if self.v_bth:
            q, k, v, g, initial = saved
            B, H, T, V = v.shape
            self.inputs = q, k, head_layout(v.reshape((B, T, H, V))), g, initial
        try:
            grads = scalar_grad(self, dy, df)
        finally:
            self.inputs = saved
        if self.v_bth:
            grads = (
                grads[:2]
                + (head_layout(grads[2].bfloat16(), True).reshape(v.shape),)
                + grads[3:]
            )
        return grads + (None, None)


def scalar_la(
    q, k, v, g, initial, scale=1.0, raw_output=False, output_bth=False, v_bth=False
):
    return ScalarAttention.apply(
        q, k, v, g, initial, scale, raw_output, output_bth, v_bth
    )


@lru_cache(None)
def _basis(M):
    period = M - 1
    src = f"""

__global__ void basis_kernel(@ARGS_DEF) {{
    @PRECALC
    int p = blockIdx.x * 256 + threadIdx.x;
    if (p < {period * M}) {{
        int row = p / {M}, col = p % {M}, pair = col / 2;
        float omega = __fmul_rn((float)pair, {2 * math.pi / period}f);
        float phase = __fmul_rn((float)row, omega);
        float z = col % 2 ? __nv_sinf(phase) : __nv_cosf(phase);
        out0_p[p] = col == 1 ? 0.f : __fmul_rn(z, col < 2 ? {period ** (-0.5)}f : {(2 / period) ** 0.5}f);
    }}
}}
basis_kernel<<<({period * M} + 255) / 256, 256>>>(@ARGS);
"""
    header = 'extern "C" __device__ float __nv_cosf(float); extern "C" __device__ float __nv_sinf(float);'
    return jt.code(
        [period, M], "float32", [], cuda_header=header, cuda_src=src
    ).stop_grad()


@lru_cache(maxsize=8)
def _zero_states(B, H, K, M, V, visible):
    # These stopped-gradient inputs are read-only by all state kernels.
    return (
        jt.zeros((B, H, K, M), dtype="float32").stop_grad(),
        jt.zeros((B, H, M, V), dtype="float32").stop_grad(),
    )


def _rotate(x, c, s, negative=False):
    pairs = x.reshape(x.shape[:-1] + [x.shape[-1] // 2, 2])
    even, odd = pairs[..., 0], pairs[..., 1]
    if negative:
        y0 = c * even + s * odd
        y1 = -s * even + c * odd
    else:
        y0 = c * even - s * odd
        y1 = s * even + c * odd
    return jt.stack([y0, y1], dim=-1).reshape(x.shape)


class _Boundary(jt.Function):
    def execute(self, x):
        self.dtype = x.dtype
        return x + 0

    def grad(self, dy):
        return dy.cast(self.dtype)


def _scalar_la(q, k, v, g, initial, C, scale=1.0, round_output=False):
    q, k, v = (_Boundary.apply(x) for x in (q, k, v))
    # Head-major contiguous chunks keep the matrix operations batched.
    B, H, T, K = q.shape
    V = v.shape[-1]
    N = T // C
    q = q.reshape((B, H, N, C, K))
    k = k.reshape((B, H, N, C, K))
    v = v.reshape((B, H, N, C, V))
    gc = g.reshape((B, H, N, C)).cumsum(-1)
    last = gc[..., -1]
    tails = (last.unsqueeze(-1) - gc).exp()
    v_tail = (v.float32() * tails.unsqueeze(-1)).cast(v.dtype)
    updates = bmm(k, v_tail, ta=True)
    states, final = affine_scan(updates, last.exp(), initial, str(q.dtype))
    inter = bmm(q, states.cast(q.dtype)) * gc.exp().unsqueeze(-1)
    scores = bmm(q, k, tb=True)
    i = jt.arange(C, dtype="int32")
    mask = i.reshape((C, 1)) >= i.reshape((1, C))
    decay = (gc.unsqueeze(-1) - gc.unsqueeze(-2)).exp()
    scores = jt.where(mask, scores * decay, 0.0).cast(q.dtype)
    intra = bmm(scores, v)
    out = (inter + intra) * scale
    if round_output:
        out = out.cast(q.dtype)
    return out.reshape((B, H, T, V)), final


def chunk_cyfa(
    q,
    k,
    v,
    g,
    delta,
    beta,
    readout,
    q_norm_weight,
    k_norm_weight,
    q_norm_eps=1e-6,
    k_norm_eps=1e-6,
    scale=None,
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
    checkpoint_level=0,
    chunk_size=64,
):
    """Chunkwise Cyclic Flow Attention.

    Inputs q, k and v use [batch, time, heads, channels]; g, delta and
    beta use [batch, time, heads]. The readout has [heads, slots-1, slots-1].
    Returns the output and, when requested, (key_state, value_state, clock).
    Persistent states and the clock are FP32. Training uses saved chunk states.
    """
    if cu_seqlens is not None:
        raise NotImplementedError("packed layout requires native segment dispatch")
    if checkpoint_level != 0:
        raise NotImplementedError("only saved-activation training is implemented")
    B, T, H, K = q.shape
    V = v.shape[-1]
    M = readout.shape[-1] + 1
    C = chunk_size
    padding = (-T) % C
    if padding:

        def pad(x):
            return jt.concat(
                [x, jt.zeros((B, padding) + tuple(x.shape[2:]), dtype=x.dtype)], dim=1
            )

        q, k, v, g, delta, beta = map(pad, (q, k, v, g, delta, beta))
    TP = T + padding
    q_norm = None
    if (
        jt.flags.no_grad
        and os.environ.get("CYFA_NATIVE_FUSED", "1") == "1"
        and M == 128
        and K == 256
        and V == 256
        and C == 64
        and str(q.dtype) == "bfloat16"
        and str(v.dtype) == "bfloat16"
        and str(q_norm_weight.dtype) == "float32"
        and use_large(q)
    ):
        from .qk_norm import query_rms_stats

        q_norm = q_norm_weight, query_rms_stats(q, q_norm_eps)
        q = q.reshape((B, H, TP, K))
    else:
        q = head_rmsnorm(q, q_norm_weight, q_norm_eps)
    k = head_rmsnorm(k, k_norm_weight, k_norm_eps)
    v_bth = (
        os.environ.get("CYFA_NATIVE_FUSED", "1") == "1"
        and M == 128
        and K == 256
        and V == 256
        and C == 64
        and str(q.dtype) == "bfloat16"
        and str(v.dtype) == "bfloat16"
        and use_large(q)
    )
    v = (
        v.reshape((B, H, TP, V))
        if v_bth
        else (head_layout(v) if str(v.dtype) == "bfloat16" else v.transpose(0, 2, 1, 3))
    )
    g, d, beta = head_gates(g.float32(), delta.float32(), beta.float32())
    if initial_state is None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if jt.flags.use_cuda and visible and "," not in visible:
            hk, hv = _zero_states(B, H, K, M, V, visible)
        else:
            hk = jt.zeros((B, H, K, M), dtype="float32").stop_grad()
            hv = jt.zeros((B, H, M, V), dtype="float32").stop_grad()
        clock = None
    else:
        hk, hv, clock = initial_state
    lam = clock_cumsum(d)
    if clock is not None:
        lam = lam + clock.unsqueeze(-1)
    fused = (
        os.environ.get("CYFA_NATIVE_FUSED", "1") == "1"
        and M == 128
        and str(q.dtype) == "bfloat16"
        and K % 128 == 0
        and V % 128 == 0
        and C == 64
    )
    if M == 128 and str(q.dtype) == "bfloat16":
        write, c, s = phase_write(lam, beta, M, compact=fused and use_large(q))
    else:
        pair = jt.arange(M // 2, dtype="int32").reshape((1, 1, 1, M // 2))
        whole = lam.floor()
        coord = ((whole.int64().unsqueeze(-1) % (M - 1)) * pair) % (M - 1)
        coord = coord.float32() + (lam - whole).unsqueeze(-1) * pair.float32()
        coord = jt.where(coord > (M - 1) * 0.5, coord - (M - 1), coord)
        phase = coord * (2 * math.pi / (M - 1))
        c, s = phase.cos(), phase.sin()
        atom_scale = (jt.where(pair == 0, 1.0, 2.0) / (M - 1)).sqrt()
        atom = jt.stack([atom_scale * c, -atom_scale * s], dim=-1).reshape(
            (B, H, TP, M)
        )
        write = (beta.unsqueeze(-1) * atom).cast(q.dtype)
    if fused:
        table = jt.matmul(readout.float32(), _basis(M))
        weights, hkt = first_readout(
            q, k, write, g, hk, c, s, table, K**-0.5 if scale is None else scale, q_norm
        )
    else:
        if K > 128 and K % 128 == 0:
            S = K // 128
            qs = (
                q.reshape((B, H, TP, S, 128))
                .transpose(0, 1, 3, 2, 4)
                .reshape((B, H * S, TP, 128))
            )
            ks = (
                k.reshape((B, H, TP, S, 128))
                .transpose(0, 1, 3, 2, 4)
                .reshape((B, H * S, TP, 128))
            )
            ws = (
                write.unsqueeze(2)
                .broadcast((B, H, S, TP, M))
                .reshape((B, H * S, TP, M))
            )
            gs = g.unsqueeze(2).broadcast((B, H, S, TP)).reshape((B, H * S, TP))
            partial, hkt = _scalar_la(
                qs,
                ks,
                ws,
                gs,
                hk.reshape((B, H * S, 128, M)),
                C,
                K**-0.5 if scale is None else scale,
                True,
            )
            raw = partial.float32().reshape((B, H, S, TP, M)).sum(2)
            hkt = hkt.reshape((B, H, K, M))
        else:
            raw, hkt = _scalar_la(
                q, k, write, g, hk, C, K**-0.5 if scale is None else scale, True
            )
        table = jt.matmul(readout.float32(), _basis(M))
        if M == 128 and str(q.dtype) == "bfloat16":
            weights = modal_readout(raw, c, s, table)
        else:
            rotated = _rotate(raw.float32(), c, s)
            table = jt.matmul(readout.float32(), _basis(M))
            z = rotated.transpose(1, 0, 2, 3).reshape((H, B * TP, M)).cast(q.dtype)
            logits = bmm(z, table.cast(q.dtype), tb=True)
            probs = jt.nn.softmax(logits, dim=-1)
            coeff = bmm(probs.cast(q.dtype), table.cast(q.dtype))
            coeff = coeff.reshape((H, B, TP, M)).transpose(1, 0, 2, 3)
            weights = _rotate(coeff, c, s, True).cast(q.dtype)
    output_bth = fused and K == 256 and V == 256 and use_large(q)
    if fused:
        out, hvt = scalar_la(
            weights, write, v, g, hv, 1.0, False, output_bth=output_bth, v_bth=v_bth
        )
    else:
        out, hvt = _scalar_la(weights, write, v, g, hv, C, 1.0, True)
    if not output_bth:
        out = (
            head_layout(out, True)
            if str(out.dtype) == "bfloat16"
            else out.transpose(0, 2, 1, 3)
        )
    if padding:
        out = out[:, :T]
    state = None
    if output_final_state:
        final_clock = clock_total(d)
        if clock is not None:
            final_clock = final_clock + clock
        state = (hkt, hvt, final_clock)
    return out, state
