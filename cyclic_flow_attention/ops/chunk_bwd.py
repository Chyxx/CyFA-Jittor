# SPDX-License-Identifier: MIT
# See LICENSE, LICENSE.fla and THIRD_PARTY_NOTICES.txt.
"""Query, key, value and decay gradients for both recurrent passes."""

import jittor as jt

from .chunk_h import state_backward
from .utils import WMMA_HEADER, _mm


def _query_key_adjoint_first(q, k, v, g, h, dy, dh, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]
    N = T // 64
    NK = K // 128
    assert V == 128 and K == 256 and T % 64 == 0
    assert (
        str(q.dtype) == "bfloat16"
        and str(k.dtype) == "bfloat16"
        and str(v.dtype) == "bfloat16"
    )
    smem = (64 * 136 + 128 * 136 + 64 * 72) * 2
    bounds = "__launch_bounds__(512,2)"
    src = f"""

__global__ void {bounds} dqdk_adjoint(@ARGS_DEF) {{
    @PRECALC
    using namespace nvcuda;
    int bh = blockIdx.z, n = blockIdx.y, bk = blockIdx.x * 128, tid = threadIdx.x, warp = tid / 32,
        lane = tid % 32;
    int cw = warp % 8, row_group = warp / 8;
    extern __shared__ __align__(32) unsigned char smem[];
    __nv_bfloat16 *left = (__nv_bfloat16 *)smem, *rh = left + 64 * 136, *rdh = rh,
                  *a = rh + 128 * 136;
    float *wq = (float *)left, *wk = (float *)rh;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> ac;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bc;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bf;
    {{
        for (int p = tid; p < 64 * 16; p += 512) {{
            int r = p / 16, c = (p % 16) * 8;
            *(int4 *)(left + r * 136 + c) =
                *(const int4 *)((const __nv_bfloat16 *)in5_p +
                                ((long long)bh * {T} + n * 64 + r) * {V} + c);
        }}
        for (int p = tid; p < 64 * 16; p += 512) {{
            int r = p / 16, c = (p % 16) * 8;
            *(int4 *)(rh + r * 136 + c) =
                *(const int4 *)((const __nv_bfloat16 *)in2_p +
                                ((long long)bh * {T} + n * 64 + r) * {V} + c);
        }}
        __syncthreads();
        if (warp < 8) {{
            int ds_row_group = warp / 4, ds_column = warp % 4;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> dsacc[2];
            for (int j = 0; j < 2; j++)
                wmma::fill_fragment(dsacc[j], 0.f);
#pragma unroll
            for (int s = 0; s < 128; s += 16) {{
                cyfa_fast_load(bc, rh + ds_column * 16 * 136 + s, 136);
#pragma unroll
                for (int j = 0; j < 2; j++) {{
                    cyfa_fast_load(af, left + (ds_row_group * 32 + j * 16) * 136 + s, 136);
                    wmma::mma_sync(dsacc[j], af, bc, dsacc[j]);
                }}
            }}
#pragma unroll
            for (int j = 0; j < 2; j++)
                for (int e = 0; e < 8; e += 2) {{
                    int r = ds_row_group * 32 + j * 16 + lane / 4 + (e % 4 >= 2 ? 8 : 0),
                        c = ds_column * 16 + (lane % 4) * 2 + (e >= 4 ? 8 : 0);
                    float gr = in3_p[bh * {T} + n * 64 + r],
                          gc0 = in3_p[bh * {T} + n * 64 + c],
                          gc1 = in3_p[bh * {T} + n * 64 + c + 1];
                    float x0 = r >= c ? dsacc[j].x[e] * exp2f(gr - gc0) * {scale}f : 0.f,
                          x1 =
                              r >= c + 1 ? dsacc[j].x[e + 1] * exp2f(gr - gc1) * {scale}f : 0.f;
                    __nv_bfloat162 pack = __floats2bfloat162_rn(x0, x1);
                    *(unsigned *)(a + r * 72 + c) = *(unsigned *)&pack;
                }}
        }}
        __syncthreads();
    }}
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dq[2], dk[2];
    for (int j = 0; j < 2; j++) {{
        wmma::fill_fragment(dq[j], 0.f);
        wmma::fill_fragment(dk[j], 0.f);
    }}
    float boundary = 0.f, gl = in3_p[bh * {T} + n * 64 + 63];
    for (int bv = 0; bv < {V}; bv += 128) {{
        for (int p = tid; p < 128 * 16; p += 512) {{
            int r = p / 16, c = (p % 16) * 8;
            *(int4 *)(rh + r * 136 + c) =
                *(const int4 *)((const __nv_bfloat16 *)in4_p +
                                (((long long)bh * {N} + n) * {K} + bk + r) *
                                    {V} +
                                bv + c);
        }}
        __syncthreads();
#pragma unroll
        for (int s = 0; s < 128; s += 16) {{
            cyfa_fast_load(bc, rh + cw * 16 * 136 + s, 136);
#pragma unroll
            for (int j = 0; j < 2; j++) {{
                cyfa_fast_load(af, left + (row_group * 32 + j * 16) * 136 + s, 136);
                wmma::mma_sync(dq[j], af, bc, dq[j]);
            }}
        }}
        __syncthreads();
        for (int p = tid; p < 64 * 16; p += 512) {{
            int r = p / 16, c = (p % 16) * 8;
            *(int4 *)(left + r * 136 + c) = *(
                const int4 *)((const __nv_bfloat16 *)in2_p +
                              ((long long)bh * {T} + n * 64 + r) * {V} + bv + c);
        }}
        for (int p = tid; p < 128 * 16; p += 512) {{
            int r = p / 16, c = (p % 16) * 8;
            int4 hd = *(const int4 *)((const __nv_bfloat16 *)in6_p +
                                      (((long long)bh * {N} + n) * {K} + bk + r) *
                                          {V} +
                                      bv + c);
            __nv_bfloat16 *hv = (__nv_bfloat16 *)(rh + r * 136 + c), *dv = (__nv_bfloat16 *)&hd;
#pragma unroll
            for (int j = 0; j < 8; j++)
                boundary += (float)hv[j] * (float)dv[j] * exp2f(gl);
            *(int4 *)(rdh + r * 136 + c) = hd;
        }}
        __syncthreads();
#pragma unroll
        for (int s = 0; s < 128; s += 16) {{
            cyfa_fast_load(bc, rdh + cw * 16 * 136 + s, 136);
#pragma unroll
            for (int j = 0; j < 2; j++) {{
                cyfa_fast_load(af, left + (row_group * 32 + j * 16) * 136 + s, 136);
                wmma::mma_sync(dk[j], af, bc, dk[j]);
            }}
        }}
        __syncthreads();
    }}
#pragma unroll
    for (int j = 0; j < 2; j++)
        for (int e = 0; e < 8; e++) {{
            int r = (row_group * 32 + j * 16) + lane / 4 + (e % 4 >= 2 ? 8 : 0),
                c = cw * 16 + (lane % 4) * 2 + (e >= 4 ? 8 : 0) + e % 2;
            float f = exp2f(in3_p[bh * {T} + n * 64 + r]) * {scale}f,
                  t = exp2f(gl - in3_p[bh * {T} + n * 64 + r]);
            dq[j].x[e] *= f;
            dk[j].x[e] *= t;
        }}
    for (int p = tid; p < 64 * 16; p += 512) {{
        int r = p / 16, c = (p % 16) * 8;
        *(int4 *)(left + r * 136 + c) =
            *(const int4 *)((const __nv_bfloat16 *)in1_p +
                            ((long long)bh * {T} + n * 64 + r) * {K} + bk + c);
    }}
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 2; j++)
        for (int e = 0; e < 8; e++) {{
            int r = row_group * 32 + j * 16 + lane / 4 + (e % 4 >= 2 ? 8 : 0),
                c = cw * 16 + (lane % 4) * 2 + (e >= 4 ? 8 : 0) + e % 2;
            boundary += dk[j].x[e] * (float)left[r * 136 + c];
        }}

#pragma unroll
    for (int s = 0; s < 64; s += 16) {{
        cyfa_fast_load(bf, left + s * 136 + cw * 16, 136);
#pragma unroll
        for (int j = 0; j < 2; j++) {{
            cyfa_fast_load(af, a + (row_group * 32 + j * 16) * 72 + s, 72);
            wmma::mma_sync(dq[j], af, bf, dq[j]);
        }}
    }}
    __syncthreads();
    for (int p = tid; p < 64 * 16; p += 512) {{
        int r = p / 16, c = (p % 16) * 8;
        *(int4 *)(rh + r * 136 + c) =
            *(const int4 *)((const __nv_bfloat16 *)in0_p +
                            ((long long)bh * {T} + n * 64 + r) * {K} + bk + c);
    }}
    __syncthreads();
#pragma unroll
    for (int s = 0; s < 64; s += 16) {{
        cyfa_fast_load(bf, rh + s * 136 + cw * 16, 136);
#pragma unroll
        for (int j = 0; j < 2; j++) {{
            cyfa_fast_load(ac, a + s * 72 + (row_group * 32 + j * 16), 72);
            wmma::mma_sync(dk[j], ac, bf, dk[j]);
        }}
    }}
#pragma unroll
    for (int j = 0; j < 2; j++) {{
        for (int e = 0; e < 8; e += 2) {{
            int r = (row_group * 32 + j * 16) + lane / 4 + (e % 4 >= 2 ? 8 : 0),
                c = cw * 16 + (lane % 4) * 2 + (e >= 4 ? 8 : 0);
            long long pos = ((long long)bh * {T} + n * 64 + r) * {K} + bk + c;
            __nv_bfloat162 qv = __floats2bfloat162_rn(dq[j].x[e], dq[j].x[e + 1]),
                           kv = __floats2bfloat162_rn(dk[j].x[e], dk[j].x[e + 1]);
            *(unsigned *)(out0_p + pos) = *(unsigned *)&qv;
            *(unsigned *)(out1_p + pos) = *(unsigned *)&kv;
        }}
        for (int rs = 0; rs < 2; rs++) {{
            int r = (row_group * 32 + j * 16) + lane / 4 + rs * 8;
            float force = 0.f;
#pragma unroll
            for (int cgroup = 0; cgroup < 2; cgroup++)
                for (int cc = 0; cc < 2; cc++) {{
                    int e = rs * 2 + cgroup * 4 + cc,
                        c = cw * 16 + (lane % 4) * 2 + cgroup * 8 + cc;
                    long long pos =
                        ((long long)bh * {T} + n * 64 + r) * {K} + bk + c;
                    force +=
                        dq[j].x[e] * (float)rh[r * 136 + c] - dk[j].x[e] * (float)left[r * 136 + c];
                }}
            force += __shfl_xor_sync(0xffffffff, force, 1);
            force += __shfl_xor_sync(0xffffffff, force, 2);
            if (lane % 4 == 0)
                out2_p[((((long long)bh * {N} + n) * {NK} + blockIdx.x) * 64 + r) *
                           8 +
                       cw] = force;
        }}
    }}
    __shared__ float bw[16];
    for (int d = 16; d > 0; d /= 2)
        boundary += __shfl_down_sync(0xffffffff, boundary, d);
    if (lane == 0)
        bw[warp] = boundary;
    __syncthreads();
    if (tid == 0) {{
        float b = 0;
        for (int i = 0; i < 16; i++)
            b += bw[i];
        out3_p[((long long)bh * {N} + n) * {NK} + blockIdx.x] = b;
    }}
}}
cudaFuncSetAttribute(dqdk_adjoint, cudaFuncAttributeMaxDynamicSharedMemorySize, {smem});
dqdk_adjoint<<<dim3({NK}, {N}, {B * H}), 512, {smem}>>>(@ARGS);
"""
    dq, dk, force, bound = jt.code(
        [q.shape, k.shape, [B, H, N, NK, 64, 8], [B, H, N, NK]],
        ["bfloat16", "bfloat16", "float32", "float32"],
        [q, k, v, g, h, dy, dh],
        cuda_header=WMMA_HEADER,
        cuda_src=src,
    )
    src = f"""

__global__ void dg_reduce(@ARGS_DEF) {{
    @PRECALC
    int row = blockIdx.x, t = threadIdx.x, lane = t % 32;
    float x = 0;
    for (int i = 0; i < {NK}; i++) {{
        for (int w = 0; w < 8; w++)
            x += in0_p[(((long long)row * {NK} + i) * 64 + t) * 8 + w];
        if (t == 63)
            x += in1_p[(long long)row * {NK} + i];
    }}
#pragma unroll
    for (int d = 1; d < 32; d *= 2) {{
        float v = __shfl_down_sync(0xffffffff, x, d);
        if (lane + d < 32)
            x += v;
    }}
    __shared__ float carry;
    if (t == 32)
        carry = x;
    __syncthreads();
    if (t < 32)
        x += carry;
    out0_p[(long long)row * 64 + t] = x;
}}
dg_reduce<<<{B * H * N}, 64>>>(@ARGS);
"""
    dg = jt.code(g.shape, "float32", [force, bound], cuda_src=src)
    return dq, dk, dg


def _query_key_adjoint_wide(q, k, v, g, h, dy, dh, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]
    N = T // 64
    NK = K // 128
    ds = _mm(dy.reshape((B, H, N, 64, V)), v.reshape((B, H, N, 64, V)), tb=True)
    smem = (64 * 136 + 128 * 136 + 64 * 72) * 2
    bounds = "__launch_bounds__(512,2)"
    src = f"""

__global__ void {bounds} dqdk_adjoint(@ARGS_DEF) {{
    @PRECALC
    using namespace nvcuda;
    int bh = blockIdx.z, n = blockIdx.y, bk = blockIdx.x * 128, tid = threadIdx.x, warp = tid / 32,
        lane = tid % 32;
    int cw = warp % 8, row_group = warp / 8;
    extern __shared__ __align__(32) unsigned char smem[];
    __nv_bfloat16 *left = (__nv_bfloat16 *)smem, *rh = left + 64 * 136, *rdh = rh,
                  *a = rh + 128 * 136;
    float *wq = (float *)left, *wk = (float *)rh;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> ac;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bc;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dq[2], dk[2];
    for (int j = 0; j < 2; j++) {{
        wmma::fill_fragment(dq[j], 0.f);
        wmma::fill_fragment(dk[j], 0.f);
    }}
    float boundary = 0.f, gl = in3_p[bh * {T} + n * 64 + 63];
    for (int bv = 0; bv < {V}; bv += 128) {{
        for (int p = tid; p < 64 * 16; p += 512) {{
            int r = p / 16, c = (p % 16) * 8;
            *(int4 *)(left + r * 136 + c) =
                *(const int4 *)((const __nv_bfloat16 *)in5_p +
                                ((long long)bh * {T} + n * 64 + r) * {V} + bv + c);
        }}
        for (int p = tid; p < 128 * 16; p += 512) {{
            int r = p / 16, c = (p % 16) * 8;
            *(int4 *)(rh + r * 136 + c) =
                *(const int4 *)((const __nv_bfloat16 *)in4_p +
                                (((long long)bh * {N} + n) * {K} + bk + r) *
                                    {V} +
                                bv + c);
        }}
        __syncthreads();
#pragma unroll
        for (int s = 0; s < 128; s += 16) {{
            cyfa_fast_load(bc, rh + cw * 16 * 136 + s, 136);
#pragma unroll
            for (int j = 0; j < 2; j++) {{
                cyfa_fast_load(af, left + (row_group * 32 + j * 16) * 136 + s, 136);
                wmma::mma_sync(dq[j], af, bc, dq[j]);
            }}
        }}
        __syncthreads();
        for (int p = tid; p < 64 * 16; p += 512) {{
            int r = p / 16, c = (p % 16) * 8;
            *(int4 *)(left + r * 136 + c) =
                *(const int4 *)((const __nv_bfloat16 *)in2_p +
                                ((long long)bh * {T} + n * 64 + r) * {V} + bv + c);
        }}
        for (int p = tid; p < 128 * 16; p += 512) {{
            int r = p / 16, c = (p % 16) * 8;
            int4 hd = *(const int4 *)((const __nv_bfloat16 *)in6_p +
                                      (((long long)bh * {N} + n) * {K} + bk + r) *
                                          {V} +
                                      bv + c);
            __nv_bfloat16 *hv = (__nv_bfloat16 *)(rh + r * 136 + c), *dv = (__nv_bfloat16 *)&hd;
#pragma unroll
            for (int j = 0; j < 8; j++)
                boundary += (float)hv[j] * (float)dv[j] * exp2f(gl);
            *(int4 *)(rdh + r * 136 + c) = hd;
        }}
        __syncthreads();
#pragma unroll
        for (int s = 0; s < 128; s += 16) {{
            cyfa_fast_load(bc, rdh + cw * 16 * 136 + s, 136);
#pragma unroll
            for (int j = 0; j < 2; j++) {{
                cyfa_fast_load(af, left + (row_group * 32 + j * 16) * 136 + s, 136);
                wmma::mma_sync(dk[j], af, bc, dk[j]);
            }}
        }}
        __syncthreads();
    }}
#pragma unroll
    for (int j = 0; j < 2; j++)
        for (int e = 0; e < 8; e++) {{
            int r = (row_group * 32 + j * 16) + lane / 4 + (e % 4 >= 2 ? 8 : 0),
                c = cw * 16 + (lane % 4) * 2 + (e >= 4 ? 8 : 0) + e % 2;
            float f = exp2f(in3_p[bh * {T} + n * 64 + r]) * {scale}f,
                  t = exp2f(gl - in3_p[bh * {T} + n * 64 + r]);
            dq[j].x[e] *= f;
            dk[j].x[e] *= t;
            boundary +=
                dk[j].x[e] *
                (float)in1_p[((long long)bh * {T} + n * 64 + r) * {K} + bk + c];
        }}
    for (int p = tid; p < 64 * 64; p += 512) {{
        int r = p / 64, c = p % 64;
        float d = in3_p[bh * {T} + n * 64 + r] - in3_p[bh * {T} + n * 64 + c];
        a[r * 72 + c] = __float2bfloat16(
            r >= c ? in7_p[((long long)bh * {N} + n) * 4096 + r * 64 + c] * exp2f(d) *
                         {scale}f
                   : 0.f);
    }}
    for (int p = tid; p < 64 * 16; p += 512) {{
        int r = p / 16, c = (p % 16) * 8;
        *(int4 *)(rh + r * 136 + c) =
            *(const int4 *)((const __nv_bfloat16 *)in1_p +
                            ((long long)bh * {T} + n * 64 + r) * {K} + bk + c);
    }}
    __syncthreads();
#pragma unroll
    for (int s = 0; s < 64; s += 16) {{
        cyfa_fast_load(bf, rh + s * 136 + cw * 16, 136);
#pragma unroll
        for (int j = 0; j < 2; j++) {{
            cyfa_fast_load(af, a + (row_group * 32 + j * 16) * 72 + s, 72);
            wmma::mma_sync(dq[j], af, bf, dq[j]);
        }}
    }}
    __syncthreads();
    for (int p = tid; p < 64 * 16; p += 512) {{
        int r = p / 16, c = (p % 16) * 8;
        *(int4 *)(rh + r * 136 + c) =
            *(const int4 *)((const __nv_bfloat16 *)in0_p +
                            ((long long)bh * {T} + n * 64 + r) * {K} + bk + c);
    }}
    __syncthreads();
#pragma unroll
    for (int s = 0; s < 64; s += 16) {{
        cyfa_fast_load(bf, rh + s * 136 + cw * 16, 136);
#pragma unroll
        for (int j = 0; j < 2; j++) {{
            cyfa_fast_load(ac, a + s * 72 + (row_group * 32 + j * 16), 72);
            wmma::mma_sync(dk[j], ac, bf, dk[j]);
        }}
    }}
#pragma unroll
    for (int j = 0; j < 2; j++) {{
        for (int e = 0; e < 8; e += 2) {{
            int r = (row_group * 32 + j * 16) + lane / 4 + (e % 4 >= 2 ? 8 : 0),
                c = cw * 16 + (lane % 4) * 2 + (e >= 4 ? 8 : 0);
            long long pos = ((long long)bh * {T} + n * 64 + r) * {K} + bk + c;
            __nv_bfloat162 qv = __floats2bfloat162_rn(dq[j].x[e], dq[j].x[e + 1]),
                           kv = __floats2bfloat162_rn(dk[j].x[e], dk[j].x[e + 1]);
            *(unsigned *)(out0_p + pos) = *(unsigned *)&qv;
            *(unsigned *)(out1_p + pos) = *(unsigned *)&kv;
        }}
        for (int rs = 0; rs < 2; rs++) {{
            int r = (row_group * 32 + j * 16) + lane / 4 + rs * 8;
            float force = 0.f;
#pragma unroll
            for (int cgroup = 0; cgroup < 2; cgroup++)
                for (int cc = 0; cc < 2; cc++) {{
                    int e = rs * 2 + cgroup * 4 + cc,
                        c = cw * 16 + (lane % 4) * 2 + cgroup * 8 + cc;
                    long long pos =
                        ((long long)bh * {T} + n * 64 + r) * {K} + bk + c;
                    force += dq[j].x[e] * (float)in0_p[pos] - dk[j].x[e] * (float)in1_p[pos];
                }}
            force += __shfl_xor_sync(0xffffffff, force, 1);
            force += __shfl_xor_sync(0xffffffff, force, 2);
            if (lane % 4 == 0)
                out2_p[((((long long)bh * {N} + n) * {NK} + blockIdx.x) * 64 + r) *
                           8 +
                       cw] = force;
        }}
    }}
    __shared__ float bw[16];
    for (int d = 16; d > 0; d /= 2)
        boundary += __shfl_down_sync(0xffffffff, boundary, d);
    if (lane == 0)
        bw[warp] = boundary;
    __syncthreads();
    if (tid == 0) {{
        float b = 0;
        for (int i = 0; i < 16; i++)
            b += bw[i];
        out3_p[((long long)bh * {N} + n) * {NK} + blockIdx.x] = b;
    }}
}}
cudaFuncSetAttribute(dqdk_adjoint, cudaFuncAttributeMaxDynamicSharedMemorySize, {smem});
dqdk_adjoint<<<dim3({NK}, {N}, {B * H}), 512, {smem}>>>(@ARGS);
"""
    dq, dk, force, bound = jt.code(
        [q.shape, k.shape, [B, H, N, NK, 64, 8], [B, H, N, NK]],
        ["bfloat16", "bfloat16", "float32", "float32"],
        [q, k, v, g, h, dy, dh, ds],
        cuda_header=WMMA_HEADER,
        cuda_src=src,
    )
    src = f"""

__global__ void dg_reduce(@ARGS_DEF) {{
    @PRECALC
    int row = blockIdx.x, t = threadIdx.x, lane = t % 32;
    float x = 0;
    for (int i = 0; i < {NK}; i++) {{
        for (int w = 0; w < 8; w++)
            x += in0_p[(((long long)row * {NK} + i) * 64 + t) * 8 + w];
        if (t == 63)
            x += in1_p[(long long)row * {NK} + i];
    }}
#pragma unroll
    for (int d = 1; d < 32; d *= 2) {{
        float v = __shfl_down_sync(0xffffffff, x, d);
        if (lane + d < 32)
            x += v;
    }}
    __shared__ float carry;
    if (t == 32)
        carry = x;
    __syncthreads();
    if (t < 32)
        x += carry;
    out0_p[(long long)row * 64 + t] = x;
}}
dg_reduce<<<{B * H * N}, 64>>>(@ARGS);
"""
    dg = jt.code(g.shape, "float32", [force, bound], cuda_src=src)
    return dq, dk, dg


def _value_adjoint_wide(q, k, g, dy, dh, scale):
    B, H, T, K = q.shape
    V = dy.shape[-1]
    N = T // 64
    NV = V // 128
    BR = 64
    NR = 4
    outpos = f"((long long)bh*{T}+t0+r)*{V}+vt*128+c"
    if T % 64 or K % 128 or V % 128 or str(q.dtype) != "bfloat16":
        raise ValueError(
            "large DV requires BF16, T divisible by 64 and K/V divisible by 128"
        )
    smem = 128 * 136 * 2 + BR * 136 * 4
    src = f"""

__global__ void __launch_bounds__(256, 2) dv_adjoint(@ARGS_DEF) {{
    @PRECALC
    using namespace nvcuda;
    int vt = blockIdx.x % {NV}, rt = blockIdx.x / {NV}, n = blockIdx.y,
        bh = blockIdx.z;
    int t0 = n * 64 + rt * {BR}, tid = threadIdx.x, warp = tid / 32;
    extern __shared__ __align__(32) unsigned char smem[];
    __nv_bfloat16 *buf = (__nv_bfloat16 *)smem, *qs = buf + 128 * 136,
                  *scores = qs + {BR} * 136;
    float *work = (float *)qs;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bc;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> av[{NR}], dv[{NR}];
    for (int r = 0; r < {NR}; r++) {{
        wmma::fill_fragment(av[r], 0.f);
        wmma::fill_fragment(dv[r], 0.f);
    }}
    for (int base = 0; base < {K}; base += 128) {{
        for (int p = tid; p < {BR} * 16; p += 256) {{
            int r = p / 16, c = (p % 16) * 8;
            *(int4 *)(qs + r * 136 + c) =
                *(const int4 *)((const __nv_bfloat16 *)in1_p +
                                ((long long)bh * {T} + t0 + r) * {K} + base + c);
        }}
        for (int p = tid; p < 64 * 16; p += 256) {{
            int r = p / 16, c = (p % 16) * 8;
            *(int4 *)(buf + r * 136 + c) =
                *(const int4 *)((const __nv_bfloat16 *)in0_p +
                                ((long long)bh * {T} + n * 64 + r) * {K} + base +
                                c);
        }}
        __syncthreads();
        if (warp < 4) {{
#pragma unroll
            for (int s = 0; s < 128; s += 16) {{
                cyfa_fast_load(bc, buf + warp * 16 * 136 + s, 136);
#pragma unroll
                for (int r = 0; r < {NR}; r++) {{
                    cyfa_fast_load(af, qs + (r * 16) * 136 + s, 136);
                    wmma::mma_sync(av[r], af, bc, av[r]);
                }}
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
        for (int s = 0; s < 128; s += 16) {{
            cyfa_fast_load(bf, buf + s * 136 + warp * 16, 136);
#pragma unroll
            for (int r = 0; r < {NR}; r++) {{
                cyfa_fast_load(af, qs + (r * 16) * 136 + s, 136);
                wmma::mma_sync(dv[r], af, bf, dv[r]);
            }}
        }}
        __syncthreads();
    }}
    if (warp < 4)
        for (int j = 0; j < {NR}; j++)
            for (int e = 0; e < 8; e += 2) {{
                int lane = tid % 32, r = j * 16 + lane / 4 + (e % 4 >= 2 ? 8 : 0),
                    c = warp * 16 + (lane % 4) * 2 + (e >= 4 ? 8 : 0);
                float g = in2_p[bh * {T} + t0 + r];
                float x0 = rt * {BR} + r <= c
                               ? av[j].x[e] * exp2f(in2_p[bh * {T} + n * 64 + c] - g) *
                                     {scale}f
                               : 0.f;
                float x1 = rt * {BR} + r <= c + 1
                               ? av[j].x[e + 1] *
                                     exp2f(in2_p[bh * {T} + n * 64 + c + 1] - g) *
                                     {scale}f
                               : 0.f;
                __nv_bfloat162 pack = __floats2bfloat162_rn(x0, x1);
                *(unsigned *)(scores + r * 72 + c) = *(unsigned *)&pack;
            }}
    __syncthreads();
    for (int r = 0; r < {NR}; r++)
        for (int e = 0; e < 8; e++) {{
            int row = r * 16 + (tid % 32) / 4 + (e % 4 >= 2 ? 8 : 0);
            dv[r].x[e] *=
                exp2f(in2_p[bh * {T} + n * 64 + 63] - in2_p[bh * {T} + t0 + row]);
        }}
    for (int p = tid; p < 64 * 16; p += 256) {{
        int r = p / 16, c = (p % 16) * 8;
        *(int4 *)(buf + r * 136 + c) =
            *(const int4 *)((const __nv_bfloat16 *)in3_p +
                            ((long long)bh * {T} + n * 64 + r) * {V} + vt * 128 +
                            c);
    }}
    __syncthreads();
#pragma unroll
    for (int s = 0; s < 64; s += 16) {{
        cyfa_fast_load(bf, buf + s * 136 + warp * 16, 136);
#pragma unroll
        for (int r = 0; r < {NR}; r++) {{
            cyfa_fast_load(af, scores + (r * 16) * 72 + s, 72);
            wmma::mma_sync(dv[r], af, bf, dv[r]);
        }}
    }}
    __syncthreads();
    for (int r = 0; r < {NR}; r++)
        wmma::store_matrix_sync(work + (r * 16) * 136 + warp * 16, dv[r], 136, wmma::mem_row_major);
    __syncthreads();
    for (int p = tid; p < {BR} * 32; p += 256) {{
        int r = p / 32, c = (p % 32) * 4;
        float4 x = *(float4 *)(work + r * 136 + c);
        long long pos = {outpos};
        __nv_bfloat162 lo = __floats2bfloat162_rn(x.x, x.y), hi = __floats2bfloat162_rn(x.z, x.w);
        *(uint2 *)(out0_p + pos) = make_uint2(*(unsigned *)&lo, *(unsigned *)&hi);
    }}
}}
cudaFuncSetAttribute(dv_adjoint, cudaFuncAttributeMaxDynamicSharedMemorySize, {smem});
dv_adjoint<<<dim3({64 // BR} *{NV}, {N}, {B * H}), 256, {smem}>>>(
    @ARGS);
"""
    return jt.code(
        dy.shape, "bfloat16", [q, k, g, dy, dh], cuda_header=WMMA_HEADER, cuda_src=src
    )


def _small_query_key_adjoint(q, k, v, g, h, dy, dh, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]
    N = T // 64
    NK = K // 128
    ds = _mm(dy.reshape((B, H, N, 64, V)), v.reshape((B, H, N, 64, V)), tb=True)
    smem = (64 * 136 + 128 * 136 + 64 * 72) * 2
    src = f"""

__global__ void dqdk_adjoint(@ARGS_DEF) {{
    @PRECALC
    using namespace nvcuda;
    int bh = blockIdx.z, n = blockIdx.y, bk = blockIdx.x * 128, tid = threadIdx.x, warp = tid / 32,
        lane = tid % 32;
    extern __shared__ __align__(32) unsigned char smem[];
    __nv_bfloat16 *left = (__nv_bfloat16 *)smem, *rh = left + 64 * 136, *rdh = rh,
                  *a = rh + 128 * 136;
    float *wq = (float *)left, *wk = (float *)rh;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> ac;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bc;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dq[4], dk[4];
    for (int j = 0; j < 4; j++) {{
        wmma::fill_fragment(dq[j], 0.f);
        wmma::fill_fragment(dk[j], 0.f);
    }}
    float boundary = 0.f, gl = in3_p[bh * {T} + n * 64 + 63];
    for (int bv = 0; bv < {V}; bv += 128) {{
        for (int p = tid; p < 64 * 16; p += 256) {{
            int r = p / 16, c = (p % 16) * 8;
            *(int4 *)(left + r * 136 + c) =
                *(const int4 *)((const __nv_bfloat16 *)in5_p +
                                ((long long)bh * {T} + n * 64 + r) * {V} + bv + c);
        }}
        for (int p = tid; p < 128 * 16; p += 256) {{
            int r = p / 16, c = (p % 16) * 8;
            *(int4 *)(rh + r * 136 + c) =
                *(const int4 *)((const __nv_bfloat16 *)in4_p +
                                (((long long)bh * {N} + n) * {K} + bk + r) *
                                    {V} +
                                bv + c);
        }}
        __syncthreads();
#pragma unroll
        for (int s = 0; s < 128; s += 16) {{
            cyfa_fast_load(bc, rh + warp * 16 * 136 + s, 136);
#pragma unroll
            for (int j = 0; j < 4; j++) {{
                cyfa_fast_load(af, left + j * 16 * 136 + s, 136);
                wmma::mma_sync(dq[j], af, bc, dq[j]);
            }}
        }}
        __syncthreads();
        for (int p = tid; p < 64 * 16; p += 256) {{
            int r = p / 16, c = (p % 16) * 8;
            *(int4 *)(left + r * 136 + c) =
                *(const int4 *)((const __nv_bfloat16 *)in2_p +
                                ((long long)bh * {T} + n * 64 + r) * {V} + bv + c);
        }}
        for (int p = tid; p < 128 * 16; p += 256) {{
            int r = p / 16, c = (p % 16) * 8;
            int4 hd = *(const int4 *)((const __nv_bfloat16 *)in6_p +
                                      (((long long)bh * {N} + n) * {K} + bk + r) *
                                          {V} +
                                      bv + c);
            __nv_bfloat16 *hv = (__nv_bfloat16 *)(rh + r * 136 + c), *dv = (__nv_bfloat16 *)&hd;
#pragma unroll
            for (int j = 0; j < 8; j++)
                boundary += (float)hv[j] * (float)dv[j] * exp2f(gl);
            *(int4 *)(rdh + r * 136 + c) = hd;
        }}
        __syncthreads();
#pragma unroll
        for (int s = 0; s < 128; s += 16) {{
            cyfa_fast_load(bc, rdh + warp * 16 * 136 + s, 136);
#pragma unroll
            for (int j = 0; j < 4; j++) {{
                cyfa_fast_load(af, left + j * 16 * 136 + s, 136);
                wmma::mma_sync(dk[j], af, bc, dk[j]);
            }}
        }}
        __syncthreads();
    }}
#pragma unroll
    for (int j = 0; j < 4; j++)
        for (int e = 0; e < 8; e++) {{
            int r = j * 16 + lane / 4 + (e % 4 >= 2 ? 8 : 0),
                c = warp * 16 + (lane % 4) * 2 + (e >= 4 ? 8 : 0) + e % 2;
            float f = exp2f(in3_p[bh * {T} + n * 64 + r]) * {scale}f,
                  t = exp2f(gl - in3_p[bh * {T} + n * 64 + r]);
            dq[j].x[e] *= f;
            dk[j].x[e] *= t;
            boundary +=
                dk[j].x[e] *
                (float)in1_p[((long long)bh * {T} + n * 64 + r) * {K} + bk + c];
        }}
    for (int p = tid; p < 64 * 64; p += 256) {{
        int r = p / 64, c = p % 64;
        float d = in3_p[bh * {T} + n * 64 + r] - in3_p[bh * {T} + n * 64 + c];
        a[r * 72 + c] = __float2bfloat16(
            r >= c ? in7_p[((long long)bh * {N} + n) * 4096 + r * 64 + c] * exp2f(d) *
                         {scale}f
                   : 0.f);
    }}
    for (int p = tid; p < 64 * 16; p += 256) {{
        int r = p / 16, c = (p % 16) * 8;
        *(int4 *)(rh + r * 136 + c) =
            *(const int4 *)((const __nv_bfloat16 *)in1_p +
                            ((long long)bh * {T} + n * 64 + r) * {K} + bk + c);
    }}
    __syncthreads();
#pragma unroll
    for (int s = 0; s < 64; s += 16) {{
        cyfa_fast_load(bf, rh + s * 136 + warp * 16, 136);
#pragma unroll
        for (int j = 0; j < 4; j++) {{
            cyfa_fast_load(af, a + j * 16 * 72 + s, 72);
            wmma::mma_sync(dq[j], af, bf, dq[j]);
        }}
    }}
    __syncthreads();
    for (int p = tid; p < 64 * 16; p += 256) {{
        int r = p / 16, c = (p % 16) * 8;
        *(int4 *)(rh + r * 136 + c) =
            *(const int4 *)((const __nv_bfloat16 *)in0_p +
                            ((long long)bh * {T} + n * 64 + r) * {K} + bk + c);
    }}
    __syncthreads();
#pragma unroll
    for (int s = 0; s < 64; s += 16) {{
        cyfa_fast_load(bf, rh + s * 136 + warp * 16, 136);
#pragma unroll
        for (int j = 0; j < 4; j++) {{
            cyfa_fast_load(ac, a + s * 72 + j * 16, 72);
            wmma::mma_sync(dk[j], ac, bf, dk[j]);
        }}
    }}
#pragma unroll
    for (int j = 0; j < 4; j++) {{
        for (int e = 0; e < 8; e += 2) {{
            int r = j * 16 + lane / 4 + (e % 4 >= 2 ? 8 : 0),
                c = warp * 16 + (lane % 4) * 2 + (e >= 4 ? 8 : 0);
            long long pos = ((long long)bh * {T} + n * 64 + r) * {K} + bk + c;
            __nv_bfloat162 qv = __floats2bfloat162_rn(dq[j].x[e], dq[j].x[e + 1]),
                           kv = __floats2bfloat162_rn(dk[j].x[e], dk[j].x[e + 1]);
            *(unsigned *)(out0_p + pos) = *(unsigned *)&qv;
            *(unsigned *)(out1_p + pos) = *(unsigned *)&kv;
        }}
        for (int rs = 0; rs < 2; rs++) {{
            int r = j * 16 + lane / 4 + rs * 8;
            float force = 0.f;
#pragma unroll
            for (int cgroup = 0; cgroup < 2; cgroup++)
                for (int cc = 0; cc < 2; cc++) {{
                    int e = rs * 2 + cgroup * 4 + cc,
                        c = warp * 16 + (lane % 4) * 2 + cgroup * 8 + cc;
                    long long pos =
                        ((long long)bh * {T} + n * 64 + r) * {K} + bk + c;
                    force += dq[j].x[e] * (float)in0_p[pos] - dk[j].x[e] * (float)in1_p[pos];
                }}
            force += __shfl_xor_sync(0xffffffff, force, 1);
            force += __shfl_xor_sync(0xffffffff, force, 2);
            if (lane % 4 == 0)
                out2_p[((((long long)bh * {N} + n) * {NK} + blockIdx.x) * 64 + r) *
                           8 +
                       warp] = force;
        }}
    }}
    __shared__ float bw[8];
    for (int d = 16; d > 0; d /= 2)
        boundary += __shfl_down_sync(0xffffffff, boundary, d);
    if (lane == 0)
        bw[warp] = boundary;
    __syncthreads();
    if (tid == 0) {{
        float b = 0;
        for (int i = 0; i < 8; i++)
            b += bw[i];
        out3_p[((long long)bh * {N} + n) * {NK} + blockIdx.x] = b;
    }}
}}
cudaFuncSetAttribute(dqdk_adjoint, cudaFuncAttributeMaxDynamicSharedMemorySize, {smem});
dqdk_adjoint<<<dim3({NK}, {N}, {B * H}), 256, {smem}>>>(@ARGS);
"""
    dq, dk, force, bound = jt.code(
        [q.shape, k.shape, [B, H, N, NK, 64, 8], [B, H, N, NK]],
        ["bfloat16", "bfloat16", "float32", "float32"],
        [q, k, v, g, h, dy, dh, ds],
        cuda_header=WMMA_HEADER,
        cuda_src=src,
    )
    src = f"""

__global__ void dg_reduce(@ARGS_DEF) {{
    @PRECALC
    int row = blockIdx.x, t = threadIdx.x, lane = t % 32;
    float x = 0;
    for (int i = 0; i < {NK}; i++) {{
        for (int w = 0; w < 8; w++)
            x += in0_p[(((long long)row * {NK} + i) * 64 + t) * 8 + w];
        if (t == 63)
            x += in1_p[(long long)row * {NK} + i];
    }}
#pragma unroll
    for (int d = 1; d < 32; d *= 2) {{
        float v = __shfl_down_sync(0xffffffff, x, d);
        if (lane + d < 32)
            x += v;
    }}
    __shared__ float carry;
    if (t == 32)
        carry = x;
    __syncthreads();
    if (t < 32)
        x += carry;
    out0_p[(long long)row * 64 + t] = x;
}}
dg_reduce<<<{B * H * N}, 64>>>(@ARGS);
"""
    dg = jt.code(g.shape, "float32", [force, bound], cuda_src=src)
    return dq, dk, dg


def value_adjoint(q, k, g, dy, dh, scale):
    from .utils import use_large

    if use_large(q):
        return _value_adjoint_wide(q, k, g, dy, dh, scale)
    B, H, T, K = q.shape
    V = dy.shape[-1]
    N = T // 64
    NV = V // 128
    BR = 32
    NR = 2
    smem = 128 * 136 * 2 + BR * 136 * 2 + BR * 72 * 2 + BR * 136 * 4
    src = f"""

__global__ void dv_adjoint(@ARGS_DEF) {{
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
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> av[{NR}], dv[{NR}];
    for (int r = 0; r < {NR}; r++) {{
        wmma::fill_fragment(av[r], 0.f);
        wmma::fill_fragment(dv[r], 0.f);
    }}
    for (int base = 0; base < {K}; base += 128) {{
        for (int p = tid; p < {BR} * 16; p += 256) {{
            int r = p / 16, c = (p % 16) * 8;
            *(int4 *)(qs + r * 136 + c) =
                *(const int4 *)((const __nv_bfloat16 *)in1_p +
                                ((long long)bh * {T} + t0 + r) * {K} + base + c);
        }}
        for (int p = tid; p < 64 * 16; p += 256) {{
            int r = p / 16, c = (p % 16) * 8;
            *(int4 *)(buf + r * 136 + c) =
                *(const int4 *)((const __nv_bfloat16 *)in0_p +
                                ((long long)bh * {T} + n * 64 + r) * {K} + base +
                                c);
        }}
        __syncthreads();
        if (warp < 4) {{
#pragma unroll
            for (int s = 0; s < 128; s += 16) {{
                cyfa_fast_load(bc, buf + warp * 16 * 136 + s, 136);
#pragma unroll
                for (int r = 0; r < {NR}; r++) {{
                    cyfa_fast_load(af, qs + r * 16 * 136 + s, 136);
                    wmma::mma_sync(av[r], af, bc, av[r]);
                }}
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
        for (int s = 0; s < 128; s += 16) {{
            cyfa_fast_load(bf, buf + s * 136 + warp * 16, 136);
#pragma unroll
            for (int r = 0; r < {NR}; r++) {{
                cyfa_fast_load(af, qs + r * 16 * 136 + s, 136);
                wmma::mma_sync(dv[r], af, bf, dv[r]);
            }}
        }}
        __syncthreads();
    }}
    if (warp < 4)
        for (int r = 0; r < {NR}; r++)
            wmma::store_matrix_sync(work + r * 16 * 136 + warp * 16, av[r], 136,
                                    wmma::mem_row_major);
    __syncthreads();
    for (int p = tid; p < {BR} * 64; p += 256) {{
        int r = p / 64, c = p % 64;
        float d = in2_p[bh * {T} + n * 64 + c] - in2_p[bh * {T} + t0 + r];
        scores[r * 72 + c] = __float2bfloat16(
            rt * {BR} + r <= c ? work[r * 136 + c] * exp2f(d) * {scale}f : 0.f);
    }}
    __syncthreads();
    for (int r = 0; r < {NR}; r++)
        wmma::store_matrix_sync(work + r * 16 * 136 + warp * 16, dv[r], 136, wmma::mem_row_major);
    __syncthreads();
    for (int p = tid; p < {BR} * 32; p += 256) {{
        int r = p / 32, c = (p % 32) * 4;
        float4 z = *(float4 *)(work + r * 136 + c);
        float f = exp2f(in2_p[bh * {T} + n * 64 + 63] - in2_p[bh * {T} + t0 + r]);
        z.x *= f;
        z.y *= f;
        z.z *= f;
        z.w *= f;
        *(float4 *)(work + r * 136 + c) = z;
    }}
    for (int p = tid; p < 64 * 16; p += 256) {{
        int r = p / 16, c = (p % 16) * 8;
        *(int4 *)(buf + r * 136 + c) =
            *(const int4 *)((const __nv_bfloat16 *)in3_p +
                            ((long long)bh * {T} + n * 64 + r) * {V} + vt * 128 +
                            c);
    }}
    __syncthreads();
    for (int r = 0; r < {NR}; r++)
        wmma::load_matrix_sync(dv[r], work + r * 16 * 136 + warp * 16, 136, wmma::mem_row_major);
#pragma unroll
    for (int s = 0; s < 64; s += 16) {{
        cyfa_fast_load(bf, buf + s * 136 + warp * 16, 136);
#pragma unroll
        for (int r = 0; r < {NR}; r++) {{
            cyfa_fast_load(af, scores + r * 16 * 72 + s, 72);
            wmma::mma_sync(dv[r], af, bf, dv[r]);
        }}
    }}
    for (int r = 0; r < {NR}; r++)
        wmma::store_matrix_sync(work + r * 16 * 136 + warp * 16, dv[r], 136, wmma::mem_row_major);
    __syncthreads();
    for (int p = tid; p < {BR} * 32; p += 256) {{
        int r = p / 32, c = (p % 32) * 4;
        float4 x = *(float4 *)(work + r * 136 + c);
        long long pos = ((long long)bh * {T} + t0 + r) * {V} + vt * 128 + c;
        __nv_bfloat162 lo = __floats2bfloat162_rn(x.x, x.y), hi = __floats2bfloat162_rn(x.z, x.w);
        *(uint2 *)(out0_p + pos) = make_uint2(*(unsigned *)&lo, *(unsigned *)&hi);
    }}
}}
cudaFuncSetAttribute(dv_adjoint, cudaFuncAttributeMaxDynamicSharedMemorySize, {smem});
dv_adjoint<<<dim3({64 // BR} *{NV}, {N}, {B * H}), 256, {smem}>>>(
    @ARGS);
"""
    return jt.code(
        dy.shape, "bfloat16", [q, k, g, dy, dh], cuda_header=WMMA_HEADER, cuda_src=src
    )


def scalar_grad(self, dy, df):
    q, k, v, g, initial = self.inputs
    if df is None:
        df = jt.zeros_like(initial)
    dy = jt.zeros(v.shape, dtype="bfloat16") if dy is None else dy.bfloat16()
    dh, dh0 = state_backward(q, dy, self.gc, df, self.scale)
    dq, dk, dg = query_key_adjoint(q, k, v, self.gc, self.states, dy, dh, self.scale)
    dv = value_adjoint(q, k, self.gc, dy, dh, self.scale)
    return (
        dq,
        (dk if self.raw_output else dk.float32()),
        (dv.float32() if self.raw_output else dv),
        dg,
        dh0,
        None,
        None,
    )


def query_key_adjoint(q, k, v, g, h, dy, dh, scale):
    from .utils import use_large

    if use_large(q):
        if q.shape[-1] == 256 and v.shape[-1] == 128:
            return _query_key_adjoint_first(q, k, v, g, h, dy, dh, scale)
        return _query_key_adjoint_wide(q, k, v, g, h, dy, dh, scale)
    return _small_query_key_adjoint(q, k, v, g, h, dy, dh, scale)
