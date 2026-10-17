# SPDX-License-Identifier: MIT
# See LICENSE, LICENSE.fla and THIRD_PARTY_NOTICES.txt.
"""Constant-state autoregressive CyFA decoding."""

import jittor as jt

from .chunk import _basis


def fused_recurrent_cyfa(
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
):
    """Decode one token using constant-size key, value and clock states.

    Uses the same input layout and return convention as chunk_cyfa.
    This path is intended for inference with a single token per sequence.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    M = readout.shape[-1] + 1
    R = M - 1
    tracked = (q, k, v, g, delta, beta, readout, q_norm_weight, k_norm_weight)
    if initial_state is not None:
        tracked = tracked + tuple(initial_state)
    if not jt.flags.no_grad and any(not x.is_stop_grad() for x in tracked):
        raise ValueError(
            "fused_recurrent_cyfa is inference-only; use jt.no_grad() or chunk_cyfa for gradients"
        )
    if T != 1 or cu_seqlens is not None:
        raise NotImplementedError("native fused decode currently needs dense T=1")
    if M > 256 or K > 1024 or V > 1024:
        raise ValueError("decode shape exceeds block limits")
    if initial_state is None:
        hk = jt.zeros((B, H, K, M))
        hv = jt.zeros((B, H, M, V))
        clock = jt.zeros((B, H))
    else:
        hk, hv, clock = initial_state
    table = jt.matmul(readout.float32(), _basis(M))
    scale = K**-0.5 if scale is None else scale
    src = f"""

__global__ void decode(@ARGS_DEF) {{
    @PRECALC
    int bh = blockIdx.x, h = bh % {H}, tid = threadIdx.x;
    __shared__ float sq[{K}], sk[{K}], raw[{M}], z[{M}], prob[256],
        weights[{M}], reduce[256];
    float q2 = 0.f, k2 = 0.f;
    for (int d = tid; d < {K}; d += 256) {{
        float qv = (float)in0_p[bh * {K} + d], kv = (float)in1_p[bh * {K} + d];
        q2 += qv * qv;
        k2 += kv * kv;
    }}
    reduce[tid] = q2;
    __syncthreads();
    for (int s = 128; s > 0; s /= 2) {{
        if (tid < s)
            reduce[tid] += reduce[tid + s];
        __syncthreads();
    }}
    float qr = rsqrtf(reduce[0] / {K} + {q_norm_eps});
    __syncthreads();
    reduce[tid] = k2;
    __syncthreads();
    for (int s = 128; s > 0; s /= 2) {{
        if (tid < s)
            reduce[tid] += reduce[tid + s];
        __syncthreads();
    }}
    float kr = rsqrtf(reduce[0] / {K} + {k_norm_eps});
    for (int d = tid; d < {K}; d += 256) {{
        sq[d] = (float)(in0_type)((float)in0_p[bh * {K} + d] * qr * (float)in7_p[d]) *
                {scale};
        sk[d] = (float)(in1_type)((float)in1_p[bh * {K} + d] * kr * (float)in8_p[d]);
    }}
    __syncthreads();
    float lam = in11_p[bh] + in4_p[bh], decay = expf(in3_p[bh]);
    if (tid == 0)
        out3_p[bh] = lam;
    if (tid < {M}) {{
        int pair = tid / 2, whole = (int)floorf(lam);
        float coord =
            (float)(((whole % {R} + {R}) % {R} * pair) % {R}) +
            (lam - floorf(lam)) * pair;
        if (coord > {R} * .5f)
            coord -= {R};
        float phase = coord * (6.283185307179586f / {R});
        float w = (float)(in0_type)(in5_p[bh] * sqrtf((pair == 0 ? 1.f : 2.f) / {R}) *
                                    (tid % 2 ? -sinf(phase) : cosf(phase)));
        float dot = 0.f;
        for (int d = 0; d < {K}; d++) {{
            long long p = ((long long)bh * {K} + d) * {M} + tid;
            float val = in9_p[p] * decay + sk[d] * w;
            out1_p[p] = val;
            dot += val * sq[d];
        }}
        raw[tid] = dot;
    }}
    __syncthreads();
    if (tid < {M}) {{
        int pair = tid / 2, whole = (int)floorf(lam), mate = tid ^ 1;
        float coord =
            (float)(((whole % {R} + {R}) % {R} * pair) % {R}) +
            (lam - floorf(lam)) * pair;
        if (coord > {R} * .5f)
            coord -= {R};
        float phase = coord * (6.283185307179586f / {R}), c = cosf(phase), s = sinf(phase);
        z[tid] = (float)(in0_type)(tid % 2 ? s * raw[mate] + c * raw[tid]
                                           : c * raw[tid] - s * raw[mate]);
    }}
    __syncthreads();
    float logit = -1.e30f;
    if (tid < {R}) {{
        logit = 0.f;
        for (int m = 0; m < {M}; m++)
            logit += z[m] *
                     (float)(in0_type)in6_p[((long long)h * {R} + tid) * {M} + m];
    }}
    reduce[tid] = logit;
    __syncthreads();
    for (int s = 128; s > 0; s /= 2) {{
        if (tid < s)
            reduce[tid] = fmaxf(reduce[tid], reduce[tid + s]);
        __syncthreads();
    }}
    float maximum = reduce[0];
    __syncthreads();
    float e = tid < {R} ? expf(logit - maximum) : 0.f;
    prob[tid] = e;
    reduce[tid] = e;
    __syncthreads();
    for (int s = 128; s > 0; s /= 2) {{
        if (tid < s)
            reduce[tid] += reduce[tid + s];
        __syncthreads();
    }}
    float denominator = reduce[0];
    __syncthreads();
    prob[tid] /= denominator;
    __syncthreads();
    if (tid < {M}) {{
        float sum = 0.f;
        for (int r = 0; r < {R}; r++)
            sum += (float)(in0_type)prob[r] *
                   (float)(in0_type)in6_p[((long long)h * {R} + r) * {M} + tid];
        raw[tid] = sum;
    }}
    __syncthreads();
    if (tid < {M}) {{
        int pair = tid / 2, whole = (int)floorf(lam), mate = tid ^ 1;
        float coord =
            (float)(((whole % {R} + {R}) % {R} * pair) % {R}) +
            (lam - floorf(lam)) * pair;
        if (coord > {R} * .5f)
            coord -= {R};
        float phase = coord * (6.283185307179586f / {R}), c = cosf(phase), s = sinf(phase);
        weights[tid] = (float)(in0_type)(tid % 2 ? -s * raw[mate] + c * raw[tid]
                                                 : c * raw[tid] + s * raw[mate]);
        z[tid] = (float)(in0_type)(in5_p[bh] * sqrtf((pair == 0 ? 1.f : 2.f) / {R}) *
                                   (tid % 2 ? -sinf(phase) : cosf(phase)));
    }}
    __syncthreads();
    for (int d = tid; d < {V}; d += 256) {{
        float sum = 0.f, vv = (float)in2_p[bh * {V} + d];
        for (int m = 0; m < {M}; m++) {{
            long long p = ((long long)bh * {M} + m) * {V} + d;
            float val = in10_p[p] * decay + z[m] * vv;
            out2_p[p] = val;
            sum += weights[m] * val;
        }}
        out0_p[bh * {V} + d] = (out0_type)sum;
    }}
}}
decode<<<{B * H}, 256>>>(@ARGS);
"""
    out, hkt, hvt, lt = jt.code(
        [v.shape, hk.shape, hv.shape, clock.shape],
        [str(v.dtype), "float32", "float32", "float32"],
        [q, k, v, g, delta, beta, table, q_norm_weight, k_norm_weight, hk, hv, clock],
        cuda_src=src,
    )
    out.stop_grad()
    hkt.stop_grad()
    hvt.stop_grad()
    lt.stop_grad()
    return out, (hkt, hvt, lt) if output_final_state else None
