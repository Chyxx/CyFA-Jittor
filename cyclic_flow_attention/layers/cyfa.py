# SPDX-License-Identifier: MIT
# See LICENSE, LICENSE.fla and THIRD_PARTY_NOTICES.txt.
"""Cyclic Flow Attention layer and its projection/normalization components."""

import math

import jittor as jt
import numpy as np
from jittor import nn

from ..ops import chunk_cyfa, fused_recurrent_cyfa
from ..ops.utils import _mm


class BiasLinear(jt.Function):
    def execute(self, x, weight, bias):
        self.x, self.weight, self.bias = x, weight, bias
        xf = x.reshape((-1, x.shape[-1]))
        y = _mm(xf, weight, tb=True, dtype="float32")
        return (
            (y + bias.float32())
            .cast(x.dtype)
            .reshape(list(x.shape[:-1]) + [weight.shape[0]])
        )

    def grad(self, dy):
        x, w, bias = self.x, self.weight, self.bias
        d = dy.cast(x.dtype).reshape((-1, w.shape[0]))
        xf = x.reshape((-1, x.shape[-1]))
        dx = _mm(d, w, dtype=str(x.dtype)).reshape(x.shape)
        dw = _mm(d, xf, ta=True, dtype=str(w.dtype))
        db = d.float32().sum(0).cast(bias.dtype)
        return dx, dw, db


def bias_linear(x, weight, bias):
    return BiasLinear.apply(x, weight, bias)


class _BiasAdd(jt.Function):
    def execute(self, x, bias):
        self.bias_dtype = bias.dtype
        return x + bias

    def grad(self, dy):
        dims = tuple(range(dy.ndim - 1))
        return dy, dy.float32().sum(dims).cast(self.bias_dtype)


class Linear(nn.Linear):
    def execute(self, x):
        if (
            self.bias is not None
            and str(x.dtype) == "bfloat16"
            and str(self.weight.dtype) == "bfloat16"
        ):
            return bias_linear(x, self.weight, self.bias)
        y = nn.matmul_transpose(x, self.weight)
        return _BiasAdd.apply(y, self.bias) if self.bias is not None else y


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        self.weight = jt.ones((dim,))
        self.eps = eps

    def execute(self, x, gate=None):
        if (
            jt.flags.no_grad
            and jt.flags.use_cuda
            and str(x.dtype) == "bfloat16"
            and x.shape[-1] % 8 == 0
            and (gate is None or str(gate.dtype) == "bfloat16")
        ):
            return self._inference(x, gate)
        xf = x.float32()
        y = (
            xf
            * ((xf * xf).mean(-1, keepdims=True) + self.eps).rsqrt()
            * self.weight.float32()
        )
        if gate is not None:
            y = y * gate.float32().sigmoid()
        return y.cast(x.dtype)

    def _inference(self, x, gate):
        D = x.shape[-1]
        rows = x.numel() // D
        inputs = [x, self.weight]
        if gate is not None:
            inputs.append(gate)
        src = f"""
__global__ void rms_norm(@ARGS_DEF) {{
    @PRECALC
    int row = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= {rows})
        return;
    float ss = 0.f;
#pragma unroll
    for (int d = lane * 8; d < {D}; d += 256) {{
        __align__(16) __nv_bfloat16 values[8];
        *(int4 *)values = *(const int4 *)(in0_p + (long long)row * {D} + d);
#pragma unroll
        for (int j = 0; j < 8; j++) {{
            float v = (float)values[j];
            ss += v * v;
        }}
    }}
    for (int step = 16; step; step /= 2)
        ss += __shfl_down_sync(0xffffffff, ss, step);
    float rs = rsqrtf(__shfl_sync(0xffffffff, ss, 0) / {D} + {self.eps});
#pragma unroll
    for (int d = lane * 8; d < {D}; d += 256) {{
        long long i = (long long)row * {D} + d;
        __align__(16) __nv_bfloat16 values[8], output[8], gates[8];
        *(int4 *)values = *(const int4 *)(in0_p + i);
        {"*(int4*)gates=*(const int4*)(in2_p+i);" if gate is not None else ""}
#pragma unroll
        for (int j = 0; j < 8; j++) {{
            float v = (float)values[j] * rs * (float)in1_p[d + j];
            {"v *= 1.f/(1.f+expf(-(float)gates[j]));" if gate is not None else ""}
            output[j] = __float2bfloat16(v);
        }}
        *(int4 *)(out0_p + i) = *(int4 *)output;
    }}
}}
rms_norm<<<({rows} + 3) / 4, 128>>>(@ARGS);
"""
        return jt.code(
            x.shape,
            str(x.dtype),
            inputs,
            cuda_header="#include <cuda_bf16.h>",
            cuda_src=src,
        )


class ShortConvolution(nn.Module):
    def __init__(self, dim, width, bias=False):
        self.width = width
        self.weight = jt.randn((dim, 1, width)) * 0.02
        self.bias = jt.zeros((dim,)) if bias else None

    def execute(self, x, cache=None, lengths=None):
        B, T, D = x.shape
        history = x if cache is None else jt.concat([cache, x], dim=1)
        offset = self.width - 1 if cache is None else 0
        if (
            jt.flags.no_grad
            and jt.flags.use_cuda
            and str(x.dtype) == "bfloat16"
            and D % 64 == 0
            and self.width <= 8
        ):
            y = self._inference(history, T, offset)
        else:
            weight = self.weight.float32()

            def window(i):
                # Fuse shifted reads and FP32 accumulation without copying each slice.
                return history.reindex(
                    [B, T, D], ["i0", f"i1+{i}-{offset}", "i2"], overflow_value=0
                ).float32()

            y = window(0) * weight[:, 0, 0]
            for i in range(1, self.width):
                y = y + window(i) * weight[:, 0, i]
            if self.bias is not None:
                y = y + self.bias.float32()
        if self.width == 1:
            new_cache = history[:, :0]
        elif lengths is None and history.shape[1] >= self.width - 1:
            new_cache = history[:, -(self.width - 1) :]
        else:
            ends = jt.full((B,), T, dtype="int32") if lengths is None else lengths
            positions = ends.unsqueeze(1) - offset + jt.arange(self.width - 1).int32()
            new_cache = history[
                jt.arange(B).int32().unsqueeze(1), jt.maximum(positions, 0)
            ]
            new_cache = jt.where((positions >= 0).unsqueeze(-1), new_cache, 0)
        return y.cast(x.dtype), new_cache

    def _inference(self, history, length, offset):
        """Reuse adjacent time steps in shared memory for BF16 inference."""
        B, S, D = history.shape
        tiles = (length + 15) // 16
        inputs = [history, self.weight]
        if self.bias is not None:
            inputs.append(self.bias)
        src = f"""
__global__ void short_conv(@ARGS_DEF) {{
    @PRECALC
    int channel = (blockIdx.x % {D // 64}) * 64;
    int batch_tile = blockIdx.x / {D // 64};
    int tile = batch_tile % {tiles}, batch = batch_tile / {tiles};
    int first = tile * 16;
    __shared__ float values[{16 + self.width - 1}][64];
    for (int i = threadIdx.x; i < {(16 + self.width - 1) * 8}; i += 256) {{
        int row = i / 8, d = (i % 8) * 8, pos = first + row - {offset};
        __align__(16) __nv_bfloat16 v[8];
        if (pos >= 0 && pos < {S})
            *(int4 *)v = *(const int4 *)(in0_p + ((long long)batch * {S} + pos) * {D} + channel + d);
#pragma unroll
        for (int j = 0; j < 8; j++)
            values[row][d + j] = (pos >= 0 && pos < {S}) ? (float)v[j] : 0.f;
    }}
    __syncthreads();
    int d = threadIdx.x % 64;
    float weights[{self.width}];
#pragma unroll
    for (int w = 0; w < {self.width}; w++)
        weights[w] = (float)in1_p[(channel + d) * {self.width} + w];
#pragma unroll
    for (int local = threadIdx.x / 64; local < 16; local += 4) {{
        if (first + local < {length}) {{
            float acc = values[local][d] * weights[0];
#pragma unroll
            for (int w = 1; w < {self.width}; w++)
                acc += values[local + w][d] * weights[w];
            {"acc += (float)in2_p[channel + d];" if self.bias is not None else ""}
            out0_p[((long long)batch * {length} + first + local) * {D} + channel + d] = (out0_type)acc;
        }}
    }}
}}
short_conv<<<{D // 64 * B * tiles}, 256>>>(@ARGS);
"""
        return jt.code(
            [B, length, D],
            "bfloat16",
            inputs,
            cuda_header="#include <cuda_bf16.h>",
            cuda_src=src,
        )


class CyclicFlowAttention(nn.Module):
    """CyFA attention with projections, short convolutions and a gated readout.

    execute returns (hidden_states, cache). The cache contains recurrent
    states and the short-convolution histories when use_cache is enabled.
    """

    def __init__(
        self,
        hidden_size=1024,
        num_heads=4,
        head_dim=256,
        expand_k=1.0,
        expand_v=1.0,
        num_slots=128,
        use_short_conv=True,
        conv_size=4,
        conv_bias=False,
        norm_eps=1e-6,
        chunk_size=64,
        **kwargs,
    ):
        self.num_heads, self.head_k_dim, self.head_v_dim = (
            num_heads,
            int(head_dim * expand_k),
            int(head_dim * expand_v),
        )
        self.key_dim, self.value_dim = (
            num_heads * self.head_k_dim,
            num_heads * self.head_v_dim,
        )
        self.num_slots, self.chunk_size = num_slots, chunk_size
        self.q_proj = Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = Linear(hidden_size, self.value_dim, bias=False)
        self.a_proj = Linear(hidden_size, num_heads, bias=False)
        self.b_proj = Linear(hidden_size, num_heads, bias=True)
        self.d_proj = Linear(hidden_size, num_heads, bias=True)
        dt = (
            jt.rand((num_heads,)) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).exp()
        self.dt_bias = dt + jt.log(1.0 - (-dt).exp())
        self.A_log = jt.log(jt.rand((num_heads,)) * 16.0 + 1e-3)
        self.readout = jt.array(
            np.broadcast_to(
                np.eye(num_slots - 1, dtype=np.float32),
                (num_heads, num_slots - 1, num_slots - 1),
            ).copy()
        )
        self.q_norm = RMSNorm(self.head_k_dim, norm_eps)
        self.k_norm = RMSNorm(self.head_k_dim, norm_eps)
        self.use_short_conv = use_short_conv
        if use_short_conv:
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, conv_bias)
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, conv_bias)
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, conv_bias)
        self.g_proj = nn.Sequential(
            Linear(hidden_size, self.head_v_dim, bias=False),
            Linear(self.head_v_dim, self.value_dim, bias=True),
        )
        self.o_norm = RMSNorm(self.head_v_dim, norm_eps)
        self.o_proj = Linear(self.value_dim, hidden_size, bias=False)

    def execute(self, x, cache=None, use_cache=False, attention_mask=None):
        B, T, _ = x.shape
        H = self.num_heads
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        lengths = None
        if attention_mask is not None:
            # Right-padding is neutral for both the recurrence and convolution.
            mask = attention_mask.unsqueeze(-1)
            q, k, v = (value * mask.cast(value.dtype) for value in (q, k, v))
            lengths = attention_mask.int32().sum(1)
        conv_cache = (
            (None, None, None)
            if cache is None
            else cache.get("conv_state", (None, None, None))
        )
        if self.use_short_conv:
            q, cq = self.q_conv1d(q, conv_cache[0], lengths)
            k, ck = self.k_conv1d(k, conv_cache[1], lengths)
            v, cv = self.v_conv1d(v, conv_cache[2], lengths)
        else:
            cq, ck, cv = None, None, None
        q = q.reshape((B, T, H, self.head_k_dim))
        k = k.reshape(q.shape)
        v = v.reshape((B, T, H, self.head_v_dim))
        a = self.a_proj(x).float32() + self.dt_bias.float32()
        g = -self.A_log.float32().exp() * nn.softplus(a)
        delta = self.d_proj(x.detach()).float32().sigmoid()
        beta = self.b_proj(x).float32().sigmoid()
        if attention_mask is not None:
            valid = attention_mask.unsqueeze(-1)
            g, delta, beta = (jt.where(valid, value, 0.0) for value in (g, delta, beta))
        kwargs = dict(
            q=q,
            k=k,
            v=v,
            g=g,
            delta=delta,
            beta=beta,
            readout=self.readout,
            q_norm_weight=self.q_norm.weight.float32(),
            k_norm_weight=self.k_norm.weight.float32(),
            q_norm_eps=self.q_norm.eps,
            k_norm_eps=self.k_norm.eps,
            initial_state=None if cache is None else cache["recurrent_state"],
            output_final_state=use_cache,
        )
        if not self.is_training() and T == 1 and jt.flags.no_grad:
            o, state = fused_recurrent_cyfa(**kwargs)
        else:
            o, state = chunk_cyfa(**kwargs, chunk_size=self.chunk_size)
        gate = self.g_proj(x).reshape((B, T, H, self.head_v_dim))
        o = self.o_norm(o * (self.num_slots - 1), gate).reshape((B, T, self.value_dim))
        o = self.o_proj(o)
        new_cache = (
            {"recurrent_state": state, "conv_state": (cq, ck, cv)}
            if use_cache
            else None
        )
        return o, new_cache
