# SPDX-License-Identifier: MIT
# See LICENSE, LICENSE.fla and THIRD_PARTY_NOTICES.txt.
"""Headwise RMS normalization and its backward pass."""

import jittor as jt


def _fixed_weight_sum(dw):
    # Native jt.code reduction: fixed row assignment and fixed summation order.
    G, D = dw.shape
    P = (G + 127) // 128
    partial = jt.code(
        (P, D),
        "float32",
        [dw],
        cuda_src=f"""

__global__ void norm_weight_part(@ARGS_DEF) {{
    @PRECALC
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32, col = blockIdx.x * 32 + lane;
    float acc = 0.f;
    __shared__ float sums[8][32];
#pragma unroll
    for (int i = 0; i < 16; i++) {{
        int row = blockIdx.y * 128 + warp + i * 8;
        if (row < {G} && col < {D})
            acc += in0_p[(long long)row * {D} + col];
    }}
    sums[warp][lane] = acc;
    __syncthreads();
    if (warp == 0 && col < {D}) {{
        float total = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++)
            total += sums[i][lane];
        out0_p[(long long)blockIdx.y * {D} + col] = total;
    }}
}}
norm_weight_part<<<dim3(({D} + 31) / 32, {P}), 256>>>(@ARGS);
""",
    )
    return jt.code(
        (D,),
        "float32",
        [partial],
        cuda_src=f"""

__global__ void norm_weight_final(@ARGS_DEF) {{
    @PRECALC
    int col = blockIdx.x * 256 + threadIdx.x;
    if (col >= {D})
        return;
    float total = 0.f;
#pragma unroll 1
    for (int i = 0; i < {P}; i++)
        total += in0_p[(long long)i * {D} + col];
    out0_p[col] = total;
}}
norm_weight_final<<<({D} + 255) / 256, 256>>>(@ARGS);
""",
    )


class VectorNorm(jt.Function):
    def execute(self, x, w, eps):
        B, T, H, D = x.shape
        R = B * T * H
        self.saved = x, w
        self.dims = B, T, H, D
        src = f"""

__global__ void vector_norm(@ARGS_DEF) {{
    @PRECALC
    int lane = threadIdx.x % 32, row = blockIdx.x * 4 + threadIdx.x / 32;
    if (row >= {R})
        return;
    int t = row % {T}, bh = row / {T}, b = bh / {H}, h = bh % {H};
    long long base =
        ((long long)b * {T} * {H} + t * {H} + h) * {D} + lane * 8;
    __align__(16) __nv_bfloat16 xv[8], yv[8];
    *(int4 *)xv = *(const int4 *)(in0_p + base);
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < 8; j++) {{
        float z = (float)xv[j];
        ss += z * z;
    }}
    for (int n = 16; n; n /= 2)
        ss += __shfl_down_sync(0xffffffff, ss, n);
    float rs = rsqrtf(__shfl_sync(0xffffffff, ss, 0) / 256.f + {eps});
    if (lane == 0)
        out1_p[row] = rs;
#pragma unroll
    for (int j = 0; j < 8; j++)
        yv[j] = __float2bfloat16((float)xv[j] * rs * in1_p[lane * 8 + j]);
    *(int4 *)(out0_p + (long long)row * {D} + lane * 8) = *(int4 *)yv;
}}
vector_norm<<<({R} + 3) / 4, 128>>>(@ARGS);
"""
        y, self.rs = jt.code(
            [[B, H, T, D], [B, H, T]],
            [str(x.dtype), "float32"],
            [x, w],
            cuda_header="#include <cuda_bf16.h>",
            cuda_src=src,
        )
        return y

    def grad(self, dy):
        B, T, H, D = self.dims
        R = B * H * T
        G = (R + 15) // 16
        x, w = self.saved
        src = f"""

__global__ void vector_norm_grad_swizzled(@ARGS_DEF) {{
    @PRECALC
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32, row = blockIdx.x * 16 + warp;
    __shared__ float part[16][256];
    float dot = 0.f, rs = 0.f;
    __align__(16) __nv_bfloat16 xv[8], dv[8], dxv[8];
    long long base = 0;
    if (row < {R}) {{
        int t = row % {T}, bh = row / {T}, b = bh / {H},
            h = bh % {H};
        base = ((long long)b * {T} * {H} + t * {H} + h) * {D} +
               lane * 8;
        *(int4 *)xv = *(const int4 *)(in0_p + base);
        *(int4 *)dv = *(const int4 *)(in3_p + (long long)row * {D} + lane * 8);
        rs = in2_p[row];
#pragma unroll
        for (int j = 0; j < 8; j++)
            dot += (float)dv[j] * in1_p[lane * 8 + j] * (float)xv[j];
        for (int n = 16; n; n /= 2)
            dot += __shfl_down_sync(0xffffffff, dot, n);
        dot = __shfl_sync(0xffffffff, dot, 0) * rs * rs / 256.f;
    }}
#pragma unroll
    for (int j = 0; j < 8; j++) {{
        float xx = row < {R} ? (float)xv[j] : 0.f,
              dd = row < {R} ? (float)dv[j] : 0.f;
        dxv[j] = __float2bfloat16((dd * in1_p[lane * 8 + j] - xx * dot) * rs);
        part[warp][(lane * 8 + j) ^ ((lane * 8 + j) >> 5)] = xx * rs * dd;
    }}
    if (row < {R})
        *(int4 *)(out0_p + base) = *(int4 *)dxv;
    __syncthreads();
    if (threadIdx.x < 256) {{
        float sum = 0.f;
#pragma unroll
        for (int i = 0; i < 16; i++)
            sum += part[i][threadIdx.x ^ (threadIdx.x >> 5)];
        out1_p[blockIdx.x * 256 + threadIdx.x] = sum;
    }}
}}
vector_norm_grad_swizzled<<<{G}, 512>>>(@ARGS);
"""
        dx, dw = jt.code(
            [x.shape, [G, D]],
            [str(x.dtype), "float32"],
            [x, w, self.rs, dy],
            cuda_header="#include <cuda_bf16.h>",
            cuda_src=src,
        )
        return dx, _fixed_weight_sum(dw).cast(w.dtype), None


def query_rms_stats(x, eps):
    B, T, H, D = x.shape
    R = B * T * H
    assert D == 256 and str(x.dtype) == "bfloat16"
    return jt.code(
        (B, H, T),
        "float32",
        [x],
        cuda_header="#include <cuda_bf16.h>",
        cuda_src=f"""

__global__ void query_rms_stats(@ARGS_DEF) {{
    @PRECALC
    int lane = threadIdx.x % 32, row = blockIdx.x * 4 + threadIdx.x / 32;
    if (row >= {R})
        return;
    int t = row % {T}, bh = row / {T}, b = bh / {H}, h = bh % {H};
    long long base =
        ((long long)b * {T} * {H} + t * {H} + h) * {D} + lane * 8;
    __align__(16) __nv_bfloat16 xv[8];
    *(int4 *)xv = *(const int4 *)(in0_p + base);
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < 8; j++) {{
        float z = (float)xv[j];
        ss += z * z;
    }}
    for (int n = 16; n; n /= 2)
        ss += __shfl_down_sync(0xffffffff, ss, n);
    float rs = rsqrtf(__shfl_sync(0xffffffff, ss, 0) / 256.f + {eps});
    if (lane == 0)
        out0_p[row] = rs;
}}
query_rms_stats<<<({R} + 3) / 4, 128>>>(@ARGS);
""",
    )


class HeadRMSNorm(jt.Function):
    def execute(self, x, w, eps):
        B, T, H, D = x.shape
        self.x, self.w, self.dims = x, w, (B, T, H, D)
        self.eps = eps
        src = f"""

__global__ void norm(@ARGS_DEF) {{
    @PRECALC
    int row = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= {B * H * T})
        return;
    int t = row % {T}, bh = row / {T}, b = bh / {H}, h = bh % {H};
    long long base = ((long long)b * {T} * {H} + t * {H} + h) * {D};
    float ss = 0.f;
    for (int d = lane; d < {D}; d += 32) {{
        float v = (float)in0_p[base + d];
        ss += v * v;
    }}
    for (int d = 16; d > 0; d /= 2)
        ss += __shfl_down_sync(0xffffffff, ss, d);
    float rs = rsqrtf(__shfl_sync(0xffffffff, ss, 0) / {D} + {eps});
    if (lane == 0)
        out1_p[row] = rs;
    for (int d = lane; d < {D}; d += 32)
        out0_p[(long long)row * {D} + d] =
            (out0_type)((float)in0_p[base + d] * rs * in1_p[d]);
}}
norm<<<({B * H * T} + 3) / 4, 128>>>(@ARGS);
"""
        y, self.rs = jt.code(
            [[B, H, T, D], [B, H, T]], [str(x.dtype), "float32"], [x, w], cuda_src=src
        )
        return y

    def grad(self, dy):
        B, T, H, D = self.dims
        rows = B * H * T
        groups = (rows + 127) // 128
        src = f"""

__global__ void norm_grad(@ARGS_DEF) {{
    @PRECALC
    int row = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= {rows})
        return;
    int t = row % {T}, bh = row / {T}, b = bh / {H}, h = bh % {H};
    long long p = ((long long)b * {T} * {H} + t * {H} + h) * {D};
    float rs = in2_p[row], dot = 0.f;
    for (int d = lane; d < {D}; d += 32)
        dot += (float)in3_p[(long long)row * {D} + d] * in1_p[d] * (float)in0_p[p + d];
    for (int d = 16; d > 0; d /= 2)
        dot += __shfl_down_sync(0xffffffff, dot, d);
    dot = __shfl_sync(0xffffffff, dot, 0) * rs * rs / {D};
    for (int d = lane; d < {D}; d += 32) {{
        float xv = (float)in0_p[p + d], dout = (float)in3_p[(long long)row * {D} + d];
        out0_p[p + d] = (out0_type)((dout * in1_p[d] - xv * dot) * rs);
    }}
}}
__global__ void weight_grad(@ARGS_DEF) {{
    @PRECALC
    int d = blockIdx.x * 128 + threadIdx.x, group = blockIdx.y;
    if (d >= {D})
        return;
    float sum = 0.f;
    for (int row = group * 128; row < (group + 1) * 128 && row < {rows}; row++) {{
        int t = row % {T}, bh = row / {T}, b = bh / {H},
            h = bh % {H};
        long long p =
            ((long long)b * {T} * {H} + t * {H} + h) * {D} + d;
        sum += (float)in0_p[p] * in2_p[row] * (float)in3_p[(long long)row * {D} + d];
    }}
    out1_p[group * {D} + d] = sum;
}}
norm_grad<<<({rows} + 3) / 4, 128>>>(@ARGS);
weight_grad<<<dim3(({D} + 127) / 128, {groups}), 128>>>(@ARGS);
"""
        dx, dw = jt.code(
            [self.x.shape, [groups, D]],
            [str(self.x.dtype), "float32"],
            [self.x, self.w, self.rs, dy],
            cuda_src=src,
        )
        return dx, dw.sum(0).cast(self.w.dtype), None


def head_rmsnorm(x, w, eps):
    from .utils import use_large

    if x.shape[-1] == 256 and str(x.dtype) == "bfloat16" and use_large(x):
        return VectorNorm.apply(x, w, eps)
    return HeadRMSNorm.apply(x, w, eps)
