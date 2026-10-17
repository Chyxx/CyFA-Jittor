# SPDX-License-Identifier: MIT
# See LICENSE, LICENSE.fla and THIRD_PARTY_NOTICES.txt.
"""Tiled scalar-attention outputs and explicit readout epilogues."""

import jittor as jt

from .utils import WMMA_HEADER


def _output_direct(
    q,
    k,
    v,
    g,
    states,
    scale,
    raw_output,
    return_source=False,
    _rows=None,
    output_bth=False,
    v_bth=False,
):
    B, H, T, K = q.shape
    V = v.shape[-1]
    N = T // 64
    NV = V // 128
    assert not raw_output and K == 128
    BR = 64
    R = BR // 16
    smem = 128 * 136 * 2 + BR * 136 * 2 + BR * 72 * 2
    assert K % 128 == 0 and V % 128 == 0 and 64 % BR == 0
    vpos = (
        f"(((long long)(bh/{H})*{T}+n*64+r)*{H}+bh%{H})*{V}+vt*128+c"
        if v_bth
        else f"((long long)bh*{T}+n*64+r)*{V}+vt*128+c"
    )
    outpos = (
        f"(((long long)(bh/{H})*{T}+t0+row)*{H}+bh%{H})*{V}+vt*128+col"
        if output_bth
        else f"((long long)bh*{T}+t0+row)*{V}+vt*128+col"
    )
    bounds = "__launch_bounds__(256,2)"
    src = f"""

__global__ void {bounds} output_direct_r64_w256(@ARGS_DEF) {{
    @PRECALC
    using namespace nvcuda;
    int vt = blockIdx.x % {NV}, rt = blockIdx.x / {NV}, n = blockIdx.y,
        bh = blockIdx.z;
    int t0 = n * 64 + rt * {BR}, tid = threadIdx.x, warp = tid / 32;
    int row_group = warp / 8;
    warp %= 8;
    extern __shared__ __align__(32) unsigned char smem[];
    __nv_bfloat16 *buf = (__nv_bfloat16 *)smem, *qs = buf + 128 * 136,
                  *scores = qs + {BR} * 136;
    float *work = (float *)qs;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bc;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[{R}];
    for (int base = 0; base < {K}; base += 128) {{
        for (int p = tid; p < {BR} * 16; p += 256) {{
            int r = p / 16, c = (p % 16) * 8;
            *(int4 *)(qs + r * 136 + c) =
                *(const int4 *)((const __nv_bfloat16 *)in0_p +
                                ((long long)bh * {T} + t0 + r) * {K} + base + c);
        }}
        for (int p = tid; p < 64 * 16; p += 256) {{
            int r = p / 16, c = (p % 16) * 8;
            *(int4 *)(buf + r * 136 + c) =
                *(const int4 *)((const __nv_bfloat16 *)in1_p +
                                ((long long)bh * {T} + n * 64 + r) * {K} + base +
                                c);
        }}
        __syncthreads();
        if (warp < 4) {{
#pragma unroll
            for (int r = 0; r < {R}; r++)
                wmma::fill_fragment(acc[r], 0.f);
#pragma unroll
            for (int s = 0; s < 128; s += 16) {{
                cyfa_fast_load(bc, buf + warp * 16 * 136 + s, 136);
#pragma unroll
                for (int r = 0; r < {R}; r++) {{
                    cyfa_fast_load(af, qs + (row_group * 32 + r * 16) * 136 + s, 136);
                    wmma::mma_sync(acc[r], af, bc, acc[r]);
                }}
            }}
#pragma unroll
            for (int j = 0; j < {R}; j++)
                for (int e = 0; e < 8; e += 2) {{
                    int lane = tid % 32,
                        r = row_group * 32 + j * 16 + lane / 4 + (e % 4 >= 2 ? 8 : 0),
                        c = warp * 16 + (lane % 4) * 2 + (e >= 4 ? 8 : 0);
                    float gq = in3_p[bh * {T} + t0 + r],
                          g0 = in3_p[bh * {T} + n * 64 + c],
                          g1 = in3_p[bh * {T} + n * 64 + c + 1];
                    float v0 = rt * {BR} + r >= c ? acc[j].x[e] * exp2f(gq - g0) : 0.f;
                    float v1 =
                        rt * {BR} + r >= c + 1 ? acc[j].x[e + 1] * exp2f(gq - g1) : 0.f;
                    __nv_bfloat162 pack = __floats2bfloat162_rn(v0, v1);
                    *(unsigned *)(scores + r * 72 + c) = *(unsigned *)&pack;
                }}
        }}
        __syncthreads();
        for (int p = tid; p < 128 * 16; p += 256) {{
            int r = p / 16, c = (p % 16) * 8;
            *(int4 *)(buf + r * 136 + c) =
                *(const int4 *)((const __nv_bfloat16 *)in4_p +
                                (((long long)bh * {N} + n) * {K} + base + r) *
                                    {V} +
                                vt * 128 + c);
        }}
        __syncthreads();
#pragma unroll
        for (int r = 0; r < {R}; r++)
            wmma::fill_fragment(acc[r], 0.f);
#pragma unroll
        for (int s = 0; s < 128; s += 16) {{
            cyfa_fast_load(bf, buf + s * 136 + warp * 16, 136);
#pragma unroll
            for (int r = 0; r < {R}; r++) {{
                cyfa_fast_load(af, qs + (row_group * 32 + r * 16) * 136 + s, 136);
                wmma::mma_sync(acc[r], af, bf, acc[r]);
            }}
        }}
#pragma unroll
        for (int r = 0; r < {R}; r++)
            for (int e = 0; e < 8; e++) {{
                int row = row_group * 32 + r * 16 + (tid % 32) / 4 + (e % 4 >= 2 ? 8 : 0);
                acc[r].x[e] *= exp2f(in3_p[bh * {T} + t0 + row]);
            }}
        __syncthreads();
        for (int p = tid; p < 64 * 16; p += 256) {{
            int r = p / 16, c = (p % 16) * 8;
            *(int4 *)(buf + r * 136 + c) =
                *(const int4 *)((const __nv_bfloat16 *)in2_p + {vpos});
        }}
        __syncthreads();
#pragma unroll
        for (int s = 0; s < 64; s += 16) {{
            cyfa_fast_load(bf, buf + s * 136 + warp * 16, 136);
#pragma unroll
            for (int r = 0; r < {R}; r++) {{
                cyfa_fast_load(af, scores + (row_group * 32 + r * 16) * 72 + s, 72);
                wmma::mma_sync(acc[r], af, bf, acc[r]);
            }}
        }}
#pragma unroll
        for (int j = 0; j < {R}; j++)
            for (int e = 0; e < 8; e += 2) {{
                int lane = tid % 32,
                    row = row_group * 32 + j * 16 + lane / 4 + (e % 4 >= 2 ? 8 : 0),
                    col = warp * 16 + (lane % 4) * 2 + (e >= 4 ? 8 : 0);
                __nv_bfloat162 pack = __floats2bfloat162_rn(acc[j].x[e] * {scale}f,
                                                            acc[j].x[e + 1] * {scale}f);
                *(unsigned *)(out0_p + {outpos}) = *(unsigned *)&pack;
            }}
    }}
}}
cudaFuncSetAttribute(output_direct_r64_w256, cudaFuncAttributeMaxDynamicSharedMemorySize,
                     {smem});
output_direct_r64_w256<<<dim3({64 // BR} *{NV}, {N}, {B * H}), 256,
                         {smem}>>>(@ARGS);
"""
    if return_source:
        return src, WMMA_HEADER
    return jt.code(
        [B, T, H, V] if output_bth else [B, H, T, V],
        "float32" if raw_output else "bfloat16",
        [q, k, v, g, states],
        cuda_header=WMMA_HEADER,
        cuda_src=src,
    )


def _output_wide(
    q,
    k,
    v,
    g,
    states,
    scale,
    raw_output,
    return_source=False,
    _rows=None,
    _threads=512,
    output_bth=False,
    v_bth=False,
    epilogue="",
    compact_raw=False,
    query_load=None,
    kernel_name="output_tiled",
):
    if not raw_output and q.shape[-1] == 128:
        return _output_direct(
            q,
            k,
            v,
            g,
            states,
            scale,
            raw_output,
            return_source,
            _rows,
            output_bth=output_bth,
            v_bth=v_bth,
        )
    assert not (output_bth or v_bth), (
        "BTH public boundary requires direct second-output route"
    )
    B, H, T, K = q.shape
    V = v.shape[-1]
    N = T // 64
    NV = V // 128
    BR = 64
    W = _threads
    assert W in (256, 512)
    R = BR // (W // 16)
    smem = 128 * 136 * 2 + max(BR * 136 * 2 + BR * 72 * 2, BR * 136 * 4)
    assert K % 128 == 0 and V % 128 == 0 and 64 % BR == 0
    previous = (
        """
uint2 bits = *(uint2 *)(out0_p + pos);
const __nv_bfloat16 *old = (const __nv_bfloat16 *)&bits;
x.x += (float)old[0];
x.y += (float)old[1];
x.z += (float)old[2];
x.w += (float)old[3];"""
        if compact_raw
        else "float4 old=*(float4*)(out0_p+pos);x.x+=old.x;x.y+=old.y;x.z+=old.z;x.w+=old.w;"
    )
    save = (
        """
__nv_bfloat162 save_lo = __floats2bfloat162_rn(x.x, x.y), save_hi = __floats2bfloat162_rn(x.z, x.w);
*(uint2 *)(out0_p + pos) = make_uint2(*(unsigned *)&save_lo, *(unsigned *)&save_hi);"""
        if compact_raw
        else "*(float4*)(out0_p+pos)=x;"
    )
    if raw_output:
        store = f"""x.x=(float)__float2bfloat16(x.x*{scale}f);x.y=(float)__float2bfloat16(x.y*{scale}f);
        x.z=(float)__float2bfloat16(x.z*{scale}f);x.w=(float)__float2bfloat16(x.w*{scale}f);
        if(base>0){{{previous}}}{save}"""
        if epilogue:
            store += "*(float4*)(work+r*136+c)=x;"
    else:
        store = f"""
__nv_bfloat162 lo = __floats2bfloat162_rn(x.x * {scale}f, x.y *{scale}f),
               hi = __floats2bfloat162_rn(x.z * {scale}f, x.w *{scale}f);
*(uint2 *)(out0_p + pos) = make_uint2(*(unsigned *)&lo, *(unsigned *)&hi);"""
    bounds = f"__launch_bounds__({W},2)"
    if query_load is None:
        query_load = f"""
for (int p = tid; p < {BR} * 16; p += {W}) {{
    int r = p / 16, c = (p % 16) * 8;
    *(int4 *)(qs + r * 136 + c) =
        *(const int4 *)((const __nv_bfloat16 *)in0_p +
                        ((long long)bh * {T} + t0 + r) * {K} + base + c);
}}"""
    src = f"""

__global__ void {bounds} {kernel_name}(@ARGS_DEF) {{
    @PRECALC
    using namespace nvcuda;
    int vt = blockIdx.x % {NV}, rt = blockIdx.x / {NV}, n = blockIdx.y,
        bh = blockIdx.z;
    int t0 = n * 64 + rt * {BR}, tid = threadIdx.x, warp = tid / 32;
    int row_group = warp / 8;
    warp %= 8;
    extern __shared__ __align__(32) unsigned char smem[];
    __nv_bfloat16 *buf = (__nv_bfloat16 *)smem, *qs = buf + 128 * 136,
                  *scores = qs + {BR} * 136;
    float *work = (float *)qs;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bc;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[{R}];
    for (int base = 0; base < {K}; base += 128) {{
        {query_load}
        for (int p = tid; p < 64 * 16; p += {W}) {{
            int r = p / 16, c = (p % 16) * 8;
            *(int4 *)(buf + r * 136 + c) =
                *(const int4 *)((const __nv_bfloat16 *)in1_p +
                                ((long long)bh * {T} + n * 64 + r) * {K} + base +
                                c);
        }}
        __syncthreads();
        if (warp < 4) {{
#pragma unroll
            for (int r = 0; r < {R}; r++)
                wmma::fill_fragment(acc[r], 0.f);
#pragma unroll
            for (int s = 0; s < 128; s += 16) {{
                cyfa_fast_load(bc, buf + warp * 16 * 136 + s, 136);
#pragma unroll
                for (int r = 0; r < {R}; r++) {{
                    cyfa_fast_load(af, qs + (row_group * 32 + r * 16) * 136 + s, 136);
                    wmma::mma_sync(acc[r], af, bc, acc[r]);
                }}
            }}
#pragma unroll
            for (int j = 0; j < {R}; j++)
                for (int e = 0; e < 8; e += 2) {{
                    int lane = tid % 32,
                        r = row_group * 32 + j * 16 + lane / 4 + (e % 4 >= 2 ? 8 : 0),
                        c = warp * 16 + (lane % 4) * 2 + (e >= 4 ? 8 : 0);
                    float gq = in3_p[bh * {T} + t0 + r],
                          g0 = in3_p[bh * {T} + n * 64 + c],
                          g1 = in3_p[bh * {T} + n * 64 + c + 1];
                    float v0 = rt * {BR} + r >= c ? acc[j].x[e] * exp2f(gq - g0) : 0.f;
                    float v1 =
                        rt * {BR} + r >= c + 1 ? acc[j].x[e + 1] * exp2f(gq - g1) : 0.f;
                    __nv_bfloat162 pack = __floats2bfloat162_rn(v0, v1);
                    *(unsigned *)(scores + r * 72 + c) = *(unsigned *)&pack;
                }}
        }}
        __syncthreads();
        for (int p = tid; p < 128 * 16; p += {W}) {{
            int r = p / 16, c = (p % 16) * 8;
            *(int4 *)(buf + r * 136 + c) =
                *(const int4 *)((const __nv_bfloat16 *)in4_p +
                                (((long long)bh * {N} + n) * {K} + base + r) *
                                    {V} +
                                vt * 128 + c);
        }}
        __syncthreads();
#pragma unroll
        for (int r = 0; r < {R}; r++)
            wmma::fill_fragment(acc[r], 0.f);
#pragma unroll
        for (int s = 0; s < 128; s += 16) {{
            cyfa_fast_load(bf, buf + s * 136 + warp * 16, 136);
#pragma unroll
            for (int r = 0; r < {R}; r++) {{
                cyfa_fast_load(af, qs + (row_group * 32 + r * 16) * 136 + s, 136);
                wmma::mma_sync(acc[r], af, bf, acc[r]);
            }}
        }}
#pragma unroll
        for (int r = 0; r < {R}; r++)
            for (int e = 0; e < 8; e++) {{
                int row = row_group * 32 + r * 16 + (tid % 32) / 4 + (e % 4 >= 2 ? 8 : 0);
                acc[r].x[e] *= exp2f(in3_p[bh * {T} + t0 + row]);
            }}
        __syncthreads();
        for (int p = tid; p < 64 * 16; p += {W}) {{
            int r = p / 16, c = (p % 16) * 8;
            *(int4 *)(buf + r * 136 + c) =
                *(const int4 *)((const __nv_bfloat16 *)in2_p +
                                ((long long)bh * {T} + n * 64 + r) * {V} +
                                vt * 128 + c);
        }}
        __syncthreads();
#pragma unroll
        for (int s = 0; s < 64; s += 16) {{
            cyfa_fast_load(bf, buf + s * 136 + warp * 16, 136);
#pragma unroll
            for (int r = 0; r < {R}; r++) {{
                cyfa_fast_load(af, scores + (row_group * 32 + r * 16) * 72 + s, 72);
                wmma::mma_sync(acc[r], af, bf, acc[r]);
            }}
        }}
        __syncthreads();
#pragma unroll
        for (int r = 0; r < {R}; r++)
            wmma::store_matrix_sync(work + (row_group * 32 + r * 16) * 136 + warp * 16, acc[r], 136,
                                    wmma::mem_row_major);
        __syncthreads();
        for (int p = tid; p < {BR} * 32; p += {W}) {{
            int r = p / 32, c = (p % 32) * 4;
            float4 x = *(float4 *)(work + r * 136 + c);
            long long pos = ((long long)bh * {T} + t0 + r) * {V} + vt * 128 + c;
            {store}
        }}
        __syncthreads();
    }}
    {epilogue}
}}
cudaFuncSetAttribute({kernel_name}, cudaFuncAttributeMaxDynamicSharedMemorySize, {smem});
{kernel_name}<<<dim3({64 // BR} *{NV}, {N}, {B * H}), {W},
               {smem}>>>(@ARGS);
"""
    if return_source:
        return src, WMMA_HEADER
    return jt.code(
        [B, H, T, V],
        "float32" if raw_output else "bfloat16",
        [q, k, v, g, states],
        cuda_header=WMMA_HEADER,
        cuda_src=src,
    )


def output_forward(
    q,
    k,
    v,
    g,
    states,
    scale,
    raw_output,
    return_source=False,
    _rows=None,
    output_bth=False,
    v_bth=False,
    epilogue="",
    compact_raw=False,
    query_load=None,
    kernel_name="output_tiled",
):
    from .utils import use_large

    if use_large(q):
        return _output_wide(
            q,
            k,
            v,
            g,
            states,
            scale,
            raw_output,
            return_source,
            _rows,
            output_bth=output_bth,
            v_bth=v_bth,
            epilogue=epilogue,
            compact_raw=compact_raw,
            query_load=query_load,
            kernel_name=kernel_name,
        )
    assert not (output_bth or v_bth), (
        "BTH public boundary requires large second-output route"
    )
    B, H, T, K = q.shape
    V = v.shape[-1]
    N = T // 64
    NV = V // 128
    BR = _rows or 32
    R = BR // 16
    smem = 128 * 136 * 2 + BR * 136 * 2 + BR * 72 * 2 + BR * 136 * 4
    assert K % 128 == 0 and V % 128 == 0 and 64 % BR == 0
    previous = (
        """
uint2 bits = *(uint2 *)(out0_p + pos);
const __nv_bfloat16 *old = (const __nv_bfloat16 *)&bits;
x.x += (float)old[0];
x.y += (float)old[1];
x.z += (float)old[2];
x.w += (float)old[3];"""
        if compact_raw
        else "float4 old=*(float4*)(out0_p+pos);x.x+=old.x;x.y+=old.y;x.z+=old.z;x.w+=old.w;"
    )
    save = (
        """
__nv_bfloat162 save_lo = __floats2bfloat162_rn(x.x, x.y), save_hi = __floats2bfloat162_rn(x.z, x.w);
*(uint2 *)(out0_p + pos) = make_uint2(*(unsigned *)&save_lo, *(unsigned *)&save_hi);"""
        if compact_raw
        else "*(float4*)(out0_p+pos)=x;"
    )
    if raw_output:
        store = f"""x.x=(float)__float2bfloat16(x.x*{scale}f);x.y=(float)__float2bfloat16(x.y*{scale}f);
        x.z=(float)__float2bfloat16(x.z*{scale}f);x.w=(float)__float2bfloat16(x.w*{scale}f);
        if(base>0){{{previous}}}{save}"""
        if epilogue:
            store += "*(float4*)(work+r*136+c)=x;"
    else:
        store = f"""
__nv_bfloat162 lo = __floats2bfloat162_rn(x.x * {scale}f, x.y *{scale}f),
               hi = __floats2bfloat162_rn(x.z * {scale}f, x.w *{scale}f);
*(uint2 *)(out0_p + pos) = make_uint2(*(unsigned *)&lo, *(unsigned *)&hi);"""
    if query_load is None:
        query_load = f"""
for (int p = tid; p < {BR} * 16; p += 256) {{
    int r = p / 16, c = (p % 16) * 8;
    *(int4 *)(qs + r * 136 + c) =
        *(const int4 *)((const __nv_bfloat16 *)in0_p +
                        ((long long)bh * {T} + t0 + r) * {K} + base + c);
}}"""
    src = f"""

__global__ void {kernel_name}(@ARGS_DEF) {{
    @PRECALC
    using namespace nvcuda;
    int vt = blockIdx.x % {NV}, rt = blockIdx.x / {NV}, n = blockIdx.y,
        bh = blockIdx.z;
    int t0 = n * 64 + rt * {BR}, tid = threadIdx.x, warp = tid / 32;
    extern __shared__ __align__(32) unsigned char smem[];
    __nv_bfloat16 *buf = (__nv_bfloat16 *)smem, *qs = buf + 128 * 136,
                  *scores = qs + {BR} * 136;
    float *work = (float *)(scores + {BR} * 72);
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bc;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[{R}];
    for (int base = 0; base < {K}; base += 128) {{
        {query_load}
        for (int p = tid; p < 64 * 16; p += 256) {{
            int r = p / 16, c = (p % 16) * 8;
            *(int4 *)(buf + r * 136 + c) = *(
                const int4 *)((const __nv_bfloat16 *)in1_p +
                              ((long long)bh * {T} + n * 64 + r) * {K} + base + c);
        }}
        __syncthreads();
        if (warp < 4) {{
#pragma unroll
            for (int r = 0; r < {R}; r++)
                wmma::fill_fragment(acc[r], 0.f);
#pragma unroll
            for (int s = 0; s < 128; s += 16) {{
                cyfa_fast_load(bc, buf + warp * 16 * 136 + s, 136);
#pragma unroll
                for (int r = 0; r < {R}; r++) {{
                    cyfa_fast_load(af, qs + r * 16 * 136 + s, 136);
                    wmma::mma_sync(acc[r], af, bc, acc[r]);
                }}
            }}
#pragma unroll
            for (int r = 0; r < {R}; r++)
                wmma::store_matrix_sync(work + r * 16 * 136 + warp * 16, acc[r], 136,
                                        wmma::mem_row_major);
        }}
        __syncthreads();
        for (int p = tid; p < {BR} * 64; p += 256) {{
            int r = p / 64, c = p % 64;
            float d = in3_p[bh * {T} + t0 + r] - in3_p[bh * {T} + n * 64 + c];
            scores[r * 72 + c] =
                __float2bfloat16(rt * {BR} + r >= c ? work[r * 136 + c] * exp2f(d) : 0.f);
        }}
        for (int p = tid; p < 128 * 16; p += 256) {{
            int r = p / 16, c = (p % 16) * 8;
            *(int4 *)(buf + r * 136 + c) =
                *(const int4 *)((const __nv_bfloat16 *)in4_p +
                                (((long long)bh * {N} + n) * {K} + base + r) *
                                    {V} +
                                vt * 128 + c);
        }}
        __syncthreads();
#pragma unroll
        for (int r = 0; r < {R}; r++)
            wmma::fill_fragment(acc[r], 0.f);
#pragma unroll
        for (int s = 0; s < 128; s += 16) {{
            cyfa_fast_load(bf, buf + s * 136 + warp * 16, 136);
#pragma unroll
            for (int r = 0; r < {R}; r++) {{
                cyfa_fast_load(af, qs + r * 16 * 136 + s, 136);
                wmma::mma_sync(acc[r], af, bf, acc[r]);
            }}
        }}
#pragma unroll
        for (int r = 0; r < {R}; r++)
            wmma::store_matrix_sync(work + r * 16 * 136 + warp * 16, acc[r], 136,
                                    wmma::mem_row_major);
        __syncthreads();
        for (int p = tid; p < {BR} * 32; p += 256) {{
            int r = p / 32, c = (p % 32) * 4;
            float4 z = *(float4 *)(work + r * 136 + c);
            float f = exp2f(in3_p[bh * {T} + t0 + r]);
            z.x *= f;
            z.y *= f;
            z.z *= f;
            z.w *= f;
            *(float4 *)(work + r * 136 + c) = z;
        }}
        for (int p = tid; p < 64 * 16; p += 256) {{
            int r = p / 16, c = (p % 16) * 8;
            *(int4 *)(buf + r * 136 + c) =
                *(const int4 *)((const __nv_bfloat16 *)in2_p +
                                ((long long)bh * {T} + n * 64 + r) * {V} +
                                vt * 128 + c);
        }}
        __syncthreads();
#pragma unroll
        for (int r = 0; r < {R}; r++)
            wmma::load_matrix_sync(acc[r], work + r * 16 * 136 + warp * 16, 136,
                                   wmma::mem_row_major);
#pragma unroll
        for (int s = 0; s < 64; s += 16) {{
            cyfa_fast_load(bf, buf + s * 136 + warp * 16, 136);
#pragma unroll
            for (int r = 0; r < {R}; r++) {{
                cyfa_fast_load(af, scores + r * 16 * 72 + s, 72);
                wmma::mma_sync(acc[r], af, bf, acc[r]);
            }}
        }}
#pragma unroll
        for (int r = 0; r < {R}; r++)
            wmma::store_matrix_sync(work + r * 16 * 136 + warp * 16, acc[r], 136,
                                    wmma::mem_row_major);
        __syncthreads();
        for (int p = tid; p < {BR} * 32; p += 256) {{
            int r = p / 32, c = (p % 32) * 4;
            float4 x = *(float4 *)(work + r * 136 + c);
            long long pos = ((long long)bh * {T} + t0 + r) * {V} + vt * 128 + c;
            {store}
        }}
        __syncthreads();
    }}
    {epilogue}
}}
cudaFuncSetAttribute({kernel_name}, cudaFuncAttributeMaxDynamicSharedMemorySize, {smem});
{kernel_name}<<<dim3({64 // BR} *{NV}, {N}, {B * H}), 256, {smem}>>>(
    @ARGS);
"""
    if return_source:
        return src, WMMA_HEADER
    return jt.code(
        [B, H, T, V],
        "float32" if raw_output else "bfloat16",
        [q, k, v, g, states],
        cuda_header=WMMA_HEADER,
        cuda_src=src,
    )
