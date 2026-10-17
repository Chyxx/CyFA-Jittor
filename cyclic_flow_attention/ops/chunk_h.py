# SPDX-License-Identifier: MIT
# See LICENSE, LICENSE.fla and THIRD_PARTY_NOTICES.txt.
"""Chunk boundary states, with direct and pipelined accumulation."""

import jittor as jt

from .utils import WMMA_HEADER


def _state_pipelined(k, v, g, initial, scale=None, v_bth=False):
    assert len(k.shape) == 4 and len(v.shape) == 4, (
        "state core expects BHTK/BHTV tensors"
    )
    B, H, T, K = k.shape
    V = v.shape[-1]
    N = T // 64
    assert T > 0 and T % 64 == 0 and K % 64 == 0 and V % 64 == 0, (
        "state pipeline requires T, K and V divisible by64"
    )
    assert str(k.dtype) == "bfloat16" and str(v.dtype) == "bfloat16", (
        "state pipeline requires BF16 matrix inputs"
    )
    assert tuple(v.shape[:3]) == (B, H, T), (
        "state pipeline requires matching BHT dimensions"
    )
    assert tuple(g.shape) == (B, H, T) and str(g.dtype) == "float32", (
        "state pipeline requires FP32 BHT local log2 gates"
    )
    assert tuple(initial.shape) == (B, H, K, V) and str(initial.dtype) == "float32", (
        "state pipeline requires FP32 BHKV initial state"
    )
    production_forward = (
        scale is None
        and B * H >= 32
        and T >= 1024
        and (K, V) in ((256, 128), (128, 256))
    )
    RK, CV, W = (32, 64, 64) if scale is not None or V > K else (64, 32, 64)
    if (K, V) in ((256, 128), (128, 256)):
        W = 128
    if production_forward:
        RK, CV, W = 64, 64, 128
    long_first = scale is None and B * H < 32 and T >= 4096 and K == 256 and V == 128
    if long_first:
        RK, CV, W = 16, 64, 128
    optimized_loop = T >= 4096 and (K, V) in ((256, 128), (128, 256))
    index_type = "unsigned" if optimized_loop or production_forward else "int"
    outer_pragma = (
        "#pragma unroll 4"
        if production_forward
        else "#pragma unroll 2"
        if optimized_loop and not long_first
        else ""
    )
    WC = min(W // 32, CV // 16)
    WR = W // 32 // WC
    RR = RK // (WR * 16)
    CC = CV // (WC * 16)
    A = RK + 8
    D = CV + 8
    backward = scale is not None
    first = N - 1 if backward else 0
    step = -1 if backward else 1
    name = "state_pipe_reverse" if backward else "state_pipe_forward"
    UC = CV if backward else RK
    SC = D if backward else A
    uptr = "b" if backward else "a"
    MC = RK if backward else CV
    SD = A if backward else D
    cptr = "a" if backward else "b"
    NG = 64 * MC // 8 // W
    NU = 64 * UC // 8 // W
    srcptr = (
        f"in1_p+((long long)bh*{T}+nn*64+r)*{V}+bv+c"
        if backward
        else f"in0_p+((long long)bh*{T}+nn*64+r)*{K}+bk+c"
    )
    assert not (backward and v_bth), "V layout flag is a forward public boundary"
    vpos = (
        f"(((long long)(bh/{H})*{T}+nn*64+r)*{H}+bh%{H})*{V}+bv+c"
        if v_bth
        else f"((long long)bh*{T}+nn*64+r)*{V}+bv+c"
    )
    changed = (
        f"in0_p+((long long)bh*{T}+nn*64+r)*{K}+bk+c" if backward else f"in1_p+{vpos}"
    )
    value = (
        f"__float2bfloat16((float)__float2bfloat16((float)vals[j]*{scale}f)*factor)"
        if backward
        else "__float2bfloat16((float)vals[j]*factor)"
    )
    load = f"""

#pragma unroll
for (int ii = 0; ii < {NU}; ii++) {{
    {index_type} p = tid + ii * {W}, r = p / {UC // 8}, c = (p % {UC // 8}) * 8;
    unsigned addr =
        __cvta_generic_to_shared({uptr} + nextbuf * 64 * {SC} + r * {SC} + c);
    const void *source = {srcptr};
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(addr), "l"(source));
}}
asm volatile("cp.async.commit_group;");
#pragma unroll
for (int ii = 0; ii < {NG}; ii++) {{
    {index_type} p = tid + ii * {W}, r = p / {MC // 8}, c = (p % {MC // 8}) * 8;
    pref[ii] = *(const int4 *)({changed});
    pg[ii] = in2_p[bh * {T} + nn * 64 + r];
}}
plast = in2_p[bh * {T} + nn * 64 + 63];
"""
    store = f"""

#pragma unroll
for (int ii = 0; ii < {NG}; ii++) {{
    {index_type} p = tid + ii * {W}, r = p / {MC // 8}, c = (p % {MC // 8}) * 8;
    float factor = exp2f({"pg[ii]" if backward else "plast-pg[ii]"});
    __nv_bfloat16 *vals = (__nv_bfloat16 *)&pref[ii];
    for (int j = 0; j < 8; j++)
        vals[j] = {value};
    *(int4 *)({cptr} + nextbuf * 64 * {SD} + r * {SD} + c) = pref[ii];
}}
asm volatile("cp.async.wait_group 0;");
"""
    src = f"""

__global__ void {name}(@ARGS_DEF) {{
    @PRECALC
    using namespace nvcuda;
    {index_type} bk = blockIdx.x * {RK}, bv = blockIdx.y * {CV}, bh = blockIdx.z,
                tid = threadIdx.x, warp = tid / 32, lane = tid % 32, wr = warp / {WC},
                wc = warp % {WC};
    __shared__ __align__(32) __nv_bfloat16 a[2 * 64 * {A}], b[2 * 64 * {D}];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[{RR * CC}];
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bf[{CC}];
#pragma unroll
    for (int j = 0; j < {RR}; j++)
        for (int l = 0; l < {CC}; l++)
            for (int e = 0; e < 8; e++) {{
                int r = (wr * {RR} + j) * 16 + lane / 4 + (e % 4 >= 2 ? 8 : 0),
                    c = (wc * {CC} + l) * 16 + (lane % 4) * 2 + (e >= 4 ? 8 : 0) + e % 2;
                acc[j * {CC} + l].x[e] =
                    in3_p[((long long)bh * {K} + bk + r) * {V} + bv + c];
            }}
    int4 pref[{NG}];
    float pg[{NG}], plast;
    int nn = {first}, nextbuf = 0;
    {load}{store}__syncthreads();
    {outer_pragma}
    for (int it = 0; it < {N}; it++) {{
        int n = {first} + it * ({step}), cur = it % 2;
        float decay = exp2f(in2_p[bh * {T} + n * 64 + 63]);
#pragma unroll
        for (int j = 0; j < {RR}; j++)
            for (int l = 0; l < {CC}; l++)
                for (int e = 0; e < 8; e += 2) {{
                    int r = (wr * {RR} + j) * 16 + lane / 4 + (e % 4 >= 2 ? 8 : 0),
                        c = (wc * {CC} + l) * 16 + (lane % 4) * 2 + (e >= 4 ? 8 : 0);
                    __nv_bfloat162 pack = __floats2bfloat162_rn(acc[j * {CC} + l].x[e],
                                                                acc[j * {CC} + l].x[e + 1]);
                    *(unsigned *)(out0_p +
                                  (((long long)bh * {N} + n) * {K} + bk + r) *
                                      {V} +
                                  bv + c) = *(unsigned *)&pack;
                }}
        nn = n + ({step});
        nextbuf = 1 - cur;
        if (it + 1 < {N}) {{
            {load}
        }}
#pragma unroll
        for (int j = 0; j < {RR}; j++)
            for (int l = 0; l < {CC}; l++)
                for (int e = 0; e < 8; e++)
                    acc[j * {CC} + l].x[e] *= decay;
#pragma unroll
        for (int ss = 0; ss < 64; ss += 16) {{
#pragma unroll
            for (int l = 0; l < {CC}; l++)
                cyfa_fast_load(bf[l],
                               b + cur * 64 * {D} + ss * {D} +
                                   (wc * {CC} + l) * 16,
                               {D});
#pragma unroll
            for (int j = 0; j < {RR}; j++) {{
                cyfa_fast_load(af,
                               a + cur * 64 * {A} + ss * {A} +
                                   (wr * {RR} + j) * 16,
                               {A});
#pragma unroll
                for (int l = 0; l < {CC}; l++)
                    wmma::mma_sync(acc[j * {CC} + l], af, bf[l], acc[j * {CC} + l]);
            }}
        }}
        if (it + 1 < {N}) {{
            {store}
        }}
        __syncthreads();
    }}
#pragma unroll
    for (int j = 0; j < {RR}; j++)
        for (int l = 0; l < {CC}; l++)
            for (int e = 0; e < 8; e++) {{
                int r = (wr * {RR} + j) * 16 + lane / 4 + (e % 4 >= 2 ? 8 : 0),
                    c = (wc * {CC} + l) * 16 + (lane % 4) * 2 + (e >= 4 ? 8 : 0) + e % 2;
                out1_p[((long long)bh * {K} + bk + r) * {V} + bv + c] =
                    acc[j * {CC} + l].x[e];
            }}
}}
{name}<<<dim3({K // RK}, {V // CV}, {B * H}), {W}>>>(@ARGS);
"""
    return jt.code(
        [[B, H, N, K, V], initial.shape],
        ["bfloat16", "float32"],
        [k, v, g, initial],
        cuda_header=WMMA_HEADER,
        cuda_src=src,
    )


def _state_dispatch(k, v, g, initial, scale=None, v_bth=False):
    from .utils import use_prefetch

    production_forward = (
        scale is None
        and k.shape[0] * k.shape[1] >= 32
        and k.shape[2] >= 1024
        and (k.shape[-1], v.shape[-1]) in ((256, 128), (128, 256))
    )
    if use_prefetch(k) or production_forward:
        return _state_pipelined(k, v, g, initial, scale, v_bth=v_bth)
    B, H, T, K = k.shape
    V = v.shape[-1]
    N = T // 64
    assert not (scale is not None and v_bth), (
        "V layout flag is a forward public boundary"
    )
    vpos = (
        f"(((long long)(bh/{H})*{T}+n*64+r)*{H}+bh%{H})*{V}+bv+c"
        if v_bth
        else f"((long long)bh*{T}+n*64+r)*{V}+bv+c"
    )
    backward = scale is not None
    name = "state_reverse_direct" if backward else "state_direct"
    loop = f"for(int n={N}-1;n>=0;n--)" if backward else f"for(int n=0;n<{N};n++)"
    ka = (
        f"""
float f = exp2f(in2_p[bh * {T} + n * 64 + r]);
for (int j = 0; j < 8; j++)
    vals[j] = __float2bfloat16((float)__float2bfloat16((float)vals[j] * {scale}f) * f);"""
        if backward
        else ""
    )
    vb = (
        ""
        if backward
        else f"""
float f = exp2f(last - in2_p[bh * {T} + n * 64 + r]);
for (int j = 0; j < 8; j++)
    vals[j] = __float2bfloat16((float)vals[j] * f);"""
    )
    src = f"""

__global__ void {name}(@ARGS_DEF) {{
    @PRECALC
    using namespace nvcuda;
    int bk = blockIdx.x * 64, bv = blockIdx.y * 64, bh = blockIdx.z, tid = threadIdx.x,
        warp = tid / 32, lane = tid % 32, wr = warp / 4, wc = warp % 4;
    __shared__ __align__(32) __nv_bfloat16 a[64 * 72], b[64 * 72];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2];
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bf;
#pragma unroll
    for (int j = 0; j < 2; j++)
        for (int e = 0; e < 8; e++) {{
            int r = wr * 16 + j * 32 + lane / 4 + (e % 4 >= 2 ? 8 : 0),
                c = wc * 16 + (lane % 4) * 2 + (e >= 4 ? 8 : 0) + e % 2;
            acc[j].x[e] = in3_p[((long long)bh * {K} + bk + r) * {V} + bv + c];
        }}
    {loop} {{
        float last = in2_p[bh * {T} + n * 64 + 63], decay = exp2f(last);
#pragma unroll
        for (int j = 0; j < 2; j++)
            for (int e = 0; e < 8; e += 2) {{
                int r = wr * 16 + j * 32 + lane / 4 + (e % 4 >= 2 ? 8 : 0),
                    c = wc * 16 + (lane % 4) * 2 + (e >= 4 ? 8 : 0);
                __nv_bfloat162 p = __floats2bfloat162_rn(acc[j].x[e], acc[j].x[e + 1]);
                *(unsigned *)(out0_p +
                              (((long long)bh * {N} + n) * {K} + bk + r) *
                                  {V} +
                              bv + c) = *(unsigned *)&p;
            }}
        for (int p = tid; p < 64 * 8; p += 256) {{
            int r = p / 8, c = (p % 8) * 8;
            int4 data =
                *(const int4 *)((const __nv_bfloat16 *)in0_p +
                                ((long long)bh * {T} + n * 64 + r) * {K} + bk + c);
            __nv_bfloat16 *vals = (__nv_bfloat16 *)&data;
            {ka}
            *(int4 *)(a + r * 72 + c) = data;
            data = *(const int4 *)((const __nv_bfloat16 *)in1_p + {vpos});
            {vb}
            *(int4 *)(b + r * 72 + c) = data;
        }}
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 2; j++) {{
            for (int e = 0; e < 8; e++)
                acc[j].x[e] *= decay;
#pragma unroll
            for (int s = 0; s < 64; s += 16) {{
                cyfa_fast_load(af, a + s * 72 + wr * 16 + j * 32, 72);
                cyfa_fast_load(bf, b + s * 72 + wc * 16, 72);
                wmma::mma_sync(acc[j], af, bf, acc[j]);
            }}
        }}
        __syncthreads();
    }}
#pragma unroll
    for (int j = 0; j < 2; j++)
        for (int e = 0; e < 8; e++) {{
            int r = wr * 16 + j * 32 + lane / 4 + (e % 4 >= 2 ? 8 : 0),
                c = wc * 16 + (lane % 4) * 2 + (e >= 4 ? 8 : 0) + e % 2;
            out1_p[((long long)bh * {K} + bk + r) * {V} + bv + c] = acc[j].x[e];
        }}
}}
{name}<<<dim3({K // 64}, {V // 64}, {B * H}), 256>>>(@ARGS);
"""
    return jt.code(
        [[B, H, N, K, V], initial.shape],
        ["bfloat16", "float32"],
        [k, v, g, initial],
        cuda_header=WMMA_HEADER,
        cuda_src=src,
    )


def state_forward(k, v, g, initial, v_bth=False):
    return _state_dispatch(k, v, g, initial, v_bth=v_bth)


def state_backward(q, dy, g, df, scale):
    return _state_dispatch(q, dy, g, df, scale)
