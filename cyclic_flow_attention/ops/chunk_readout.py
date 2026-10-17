# SPDX-License-Identifier: MIT
# See LICENSE, LICENSE.fla and THIRD_PARTY_NOTICES.txt.
"""Fused first-pass readout and its adjoint."""

import jittor as jt

from .chunk_bwd import scalar_grad
from .chunk_h import state_forward
from .chunk_o import _output_wide, output_forward
from .utils import CLOCK_PHASE_HEADER, WMMA_HEADER, _mm, local_log2_cumsum


def _readout_grad_small(self, dw):
    B, H, T, M = self.dims
    R = M - 1
    BR = 16
    raw, c, s, table = self.raw, self.c, self.s, self.table
    dw = dw.bfloat16()
    smem = 128 * 136 * 2 + 2 * BR * 136 * 2 + 2 * BR * 136 * 4
    src = f"""

__global__ void readout_adjoint(@ARGS_DEF) {{
    @PRECALC
    using namespace nvcuda;
    int bh = blockIdx.y, h = bh % {H}, batch = bh / {H},
        t0 = blockIdx.x * {BR}, tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    extern __shared__ __align__(32) unsigned char smem[];
    __nv_bfloat16 *w = (__nv_bfloat16 *)smem, *z = w + 128 * 136, *du = z + {BR} * 136;
    float *score = (float *)(du + {BR} * 136), *dp = score + {BR} * 136;
    for (int p = tid; p < 128 * 128; p += 256) {{
        int r = p / 128, m = p % 128;
        w[r * 136 + m] =
            __float2bfloat16(r < 127 ? in3_p[(long long)h * 127 * 128 + r * 128 + m] : 0.f);
    }}
    for (int p = tid; p < {BR} * 64; p += 256) {{
        int row = p / 64, m = (p % 64) * 2, t = t0 + row;
        float ze = 0, zo = 0, de = 0, do_ = 0;
        if (t < {T}) {{
            long long q = ((long long)bh * {T} + t) * 128 + m,
                      ph = ((long long)bh * {T} + t) * 64 + m / 2;
            float re = (float)__float2bfloat16((float)in0_p[q]),
                  ro = (float)__float2bfloat16((float)in0_p[q + 1]);
            float dwe = (float)in4_p[q], dwo = (float)in4_p[q + 1], co = in1_p[ph], si = in2_p[ph];
            ze = co * re - si * ro;
            zo = si * re + co * ro;
            de = co * dwe - si * dwo;
            do_ = si * dwe + co * dwo;
            long long dst = ((long long)h * {B * T} + batch * {T} + t) * 128 + m;
            __nv_bfloat162 zv = __floats2bfloat162_rn(ze, zo);
            *(unsigned *)(out5_p + dst) = *(unsigned *)&zv;
            __nv_bfloat162 dv = __floats2bfloat162_rn(de, do_);
            *(unsigned *)(out6_p + dst) = *(unsigned *)&dv;
        }}
        z[row * 136 + m] = __float2bfloat16(ze);
        z[row * 136 + m + 1] = __float2bfloat16(zo);
        du[row * 136 + m] = __float2bfloat16(de);
        du[row * 136 + m + 1] = __float2bfloat16(do_);
    }}
    __syncthreads();
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bc;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> a0, a1;
    wmma::fill_fragment(a0, 0.f);
    wmma::fill_fragment(a1, 0.f);
#pragma unroll
    for (int k = 0; k < 128; k += 16) {{
        cyfa_fast_load(bc, w + warp * 16 * 136 + k, 136);
        cyfa_fast_load(af, z + k, 136);
        wmma::mma_sync(a0, af, bc, a0);
        cyfa_fast_load(af, du + k, 136);
        wmma::mma_sync(a1, af, bc, a1);
    }}
    wmma::store_matrix_sync(score + warp * 16, a0, 136, wmma::mem_row_major);
    wmma::store_matrix_sync(dp + warp * 16, a1, 136, wmma::mem_row_major);
    __syncthreads();
    for (int row = warp; row < {BR}; row += 8) {{
        float mx = -1.e30f;
        for (int col = lane; col < 127; col += 32)
            mx = fmaxf(mx, score[row * 136 + col]);
        for (int d = 16; d > 0; d /= 2)
            mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, d));
        mx = __shfl_sync(0xffffffff, mx, 0);
        float sum = 0.f;
        for (int col = lane; col < 128; col += 32) {{
            float x = col < 127 ? expf(score[row * 136 + col] - mx) : 0.f;
            score[row * 136 + col] = x;
            sum += x;
        }}
        for (int d = 16; d > 0; d /= 2)
            sum += __shfl_down_sync(0xffffffff, sum, d);
        sum = __shfl_sync(0xffffffff, sum, 0);
        float correction = 0.f;
        for (int col = lane; col < 128; col += 32) {{
            float p = score[row * 136 + col] / sum;
            score[row * 136 + col] = p;
            correction += p * dp[row * 136 + col];
        }}
        for (int d = 16; d > 0; d /= 2)
            correction += __shfl_down_sync(0xffffffff, correction, d);
        correction = __shfl_sync(0xffffffff, correction, 0);
        for (int col = lane; col < 128; col += 32) {{
            float p = score[row * 136 + col], dl = p * (dp[row * 136 + col] - correction);
            z[row * 136 + col] = __float2bfloat16(p);
            du[row * 136 + col] = __float2bfloat16(dl);
        }}
    }}
    __syncthreads();
    for (int p = tid; p < {BR} * 16; p += 256) {{
        int row = p / 16, col = (p % 16) * 8;
        if (t0 + row < {T}) {{
            long long dst =
                ((long long)h * {B * T} + batch * {T} + t0 + row) * 128 + col;
            *(int4 *)(out3_p + dst) = *(int4 *)(z + row * 136 + col);
            *(int4 *)(out4_p + dst) = *(int4 *)(du + row * 136 + col);
        }}
    }}
    wmma::fill_fragment(a0, 0.f);
    wmma::fill_fragment(a1, 0.f);
#pragma unroll
    for (int k = 0; k < 128; k += 16) {{
        cyfa_fast_load(bf, w + k * 136 + warp * 16, 136);
        cyfa_fast_load(af, z + k, 136);
        wmma::mma_sync(a0, af, bf, a0);
        cyfa_fast_load(af, du + k, 136);
        wmma::mma_sync(a1, af, bf, a1);
    }}
    wmma::store_matrix_sync(score + warp * 16, a0, 136, wmma::mem_row_major);
    wmma::store_matrix_sync(dp + warp * 16, a1, 136, wmma::mem_row_major);
    __syncthreads();
    for (int p = tid; p < {BR} * 64; p += 256) {{
        int row = p / 64, m = (p % 64) * 2, t = t0 + row;
        if (t < {T}) {{
            long long q = ((long long)bh * {T} + t) * 128 + m,
                      ph = ((long long)bh * {T} + t) * 64 + m / 2;
            float re = (float)__float2bfloat16((float)in0_p[q]),
                  ro = (float)__float2bfloat16((float)in0_p[q + 1]);
            float dwe = (float)in4_p[q], dwo = (float)in4_p[q + 1], co = in1_p[ph], si = in2_p[ph];
            float ue = score[row * 136 + m], uo = score[row * 136 + m + 1], ge = dp[row * 136 + m],
                  go = dp[row * 136 + m + 1];
            __nv_bfloat162 gv = __floats2bfloat162_rn(co * ge + si * go, -si * ge + co * go);
            *(unsigned *)(out0_p + q) = *(unsigned *)&gv;
            out1_p[ph] = dwe * ue + dwo * uo + ge * re + go * ro;
            out2_p[ph] = dwe * uo - dwo * ue - ge * ro + go * re;
        }}
    }}
}}
cudaFuncSetAttribute(readout_adjoint, cudaFuncAttributeMaxDynamicSharedMemorySize, {smem});
readout_adjoint<<<dim3(({T} + {BR - 1}) / {BR}, {B * H}), 256,
                  {smem}>>>(@ARGS);
"""
    dr, dc, ds, p, dl, z, du = jt.code(
        [
            raw.shape,
            c.shape,
            s.shape,
            [H, B * T, M],
            [H, B * T, M],
            [H, B * T, M],
            [H, B * T, M],
        ],
        [
            "bfloat16",
            "float32",
            "float32",
            "bfloat16",
            "bfloat16",
            "bfloat16",
            "bfloat16",
        ],
        [raw, c, s, table, dw],
        cuda_header=WMMA_HEADER,
        cuda_src=src,
    )
    S = min(64, max(1, B * T // 256))
    pp = p.reshape((H, S, B * T // S, M))
    ll = dl.reshape(pp.shape)
    uu = du.reshape((H, S, B * T // S, M))
    zz = z.reshape(uu.shape)
    dt = (_mm(pp, uu, ta=True) + _mm(ll, zz, ta=True)).sum(1)
    return dr, dc, ds, dt[:, :R, :]


def _readout_grad_wide(self, dw):
    B, H, T, M = self.dims
    R = M - 1
    BR = 32
    NR = BR // 32
    raw, c, s, table = self.raw, self.c, self.s, self.table
    dw = dw.bfloat16()
    smem = 128 * 136 * 2 + 2 * BR * 136 * 4
    src = f"""

__global__ void readout_adjoint(@ARGS_DEF) {{
    @PRECALC
    using namespace nvcuda;
    int bh = blockIdx.y, h = bh % {H}, batch = bh / {H},
        t0 = blockIdx.x * {BR}, tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    int row_group = warp / 8, cw = warp % 8;
    extern __shared__ __align__(32) unsigned char smem[];
    __nv_bfloat16 *w = (__nv_bfloat16 *)smem, *z = w + 128 * 136, *du = z + {BR} * 136;
    float *score = (float *)z, *dp = score + {BR} * 136;
    for (int p = tid; p < 128 * 128; p += 512) {{
        int r = p / 128, m = p % 128;
        w[r * 136 + m] =
            __float2bfloat16(r < 127 ? in3_p[(long long)h * 127 * 128 + r * 128 + m] : 0.f);
    }}
    for (int p = tid; p < {BR} * 64; p += 512) {{
        int row = p / 64, m = (p % 64) * 2, t = t0 + row;
        float ze = 0, zo = 0, de = 0, do_ = 0;
        if (t < {T}) {{
            long long q = ((long long)bh * {T} + t) * 128 + m,
                      ph = ((long long)bh * {T} + t) * 64 + m / 2;
            float re = (float)__float2bfloat16((float)in0_p[q]),
                  ro = (float)__float2bfloat16((float)in0_p[q + 1]);
            float dwe = (float)in4_p[q], dwo = (float)in4_p[q + 1], co = in1_p[ph], si = in2_p[ph];
            ze = co * re - si * ro;
            zo = si * re + co * ro;
            de = co * dwe - si * dwo;
            do_ = si * dwe + co * dwo;
            long long dst = ((long long)h * {B * T} + batch * {T} + t) * 128 + m;
            __nv_bfloat162 zv = __floats2bfloat162_rn(ze, zo);
            *(unsigned *)(out5_p + dst) = *(unsigned *)&zv;
            __nv_bfloat162 dv = __floats2bfloat162_rn(de, do_);
            *(unsigned *)(out6_p + dst) = *(unsigned *)&dv;
        }}
        z[row * 136 + m] = __float2bfloat16(ze);
        z[row * 136 + m + 1] = __float2bfloat16(zo);
        du[row * 136 + m] = __float2bfloat16(de);
        du[row * 136 + m + 1] = __float2bfloat16(do_);
    }}
    __syncthreads();
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bc;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> a0[{NR}], a1[{NR}];
    for (int j = 0; j < {NR}; j++) {{
        wmma::fill_fragment(a0[j], 0.f);
        wmma::fill_fragment(a1[j], 0.f);
    }}
#pragma unroll
    for (int k = 0; k < 128; k += 16) {{
        cyfa_fast_load(bc, w + cw * 16 * 136 + k, 136);
#pragma unroll
        for (int j = 0; j < {NR}; j++) {{
            cyfa_fast_load(af, z + (row_group * {BR // 2} + j * 16) * 136 + k, 136);
            wmma::mma_sync(a0[j], af, bc, a0[j]);
            cyfa_fast_load(af, du + (row_group * {BR // 2} + j * 16) * 136 + k, 136);
            wmma::mma_sync(a1[j], af, bc, a1[j]);
        }}
    }}
    __syncthreads();
    for (int j = 0; j < {NR}; j++) {{
        wmma::store_matrix_sync(score + (row_group * {BR // 2} + j * 16) * 136 + cw * 16, a0[j],
                                136, wmma::mem_row_major);
        wmma::store_matrix_sync(dp + (row_group * {BR // 2} + j * 16) * 136 + cw * 16, a1[j],
                                136, wmma::mem_row_major);
    }}
    __syncthreads();
    float pbuf[{BR // 16}][4], lbuf[{BR // 16}][4];
#pragma unroll
    for (int ii = 0; ii < {BR // 16}; ii++) {{
        int row = warp + ii * 16;
        float mx = -1.e30f;
#pragma unroll
        for (int jj = 0; jj < 4; jj++) {{
            int col = lane + jj * 32;
            if (col < 127)
                mx = fmaxf(mx, score[row * 136 + col]);
        }}
        for (int d = 16; d > 0; d /= 2)
            mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, d));
        mx = __shfl_sync(0xffffffff, mx, 0);
        float sum = 0.f, ex[4];
#pragma unroll
        for (int jj = 0; jj < 4; jj++) {{
            int col = lane + jj * 32;
            ex[jj] = col < 127 ? expf(score[row * 136 + col] - mx) : 0.f;
            sum += ex[jj];
        }}
        for (int d = 16; d > 0; d /= 2)
            sum += __shfl_down_sync(0xffffffff, sum, d);
        sum = __shfl_sync(0xffffffff, sum, 0);
        float correction = 0.f;
#pragma unroll
        for (int jj = 0; jj < 4; jj++) {{
            int col = lane + jj * 32;
            pbuf[ii][jj] = ex[jj] / sum;
            correction += pbuf[ii][jj] * dp[row * 136 + col];
        }}
        for (int d = 16; d > 0; d /= 2)
            correction += __shfl_down_sync(0xffffffff, correction, d);
        correction = __shfl_sync(0xffffffff, correction, 0);
#pragma unroll
        for (int jj = 0; jj < 4; jj++) {{
            int col = lane + jj * 32;
            lbuf[ii][jj] = pbuf[ii][jj] * (dp[row * 136 + col] - correction);
        }}
    }}
    __syncthreads();
#pragma unroll
    for (int ii = 0; ii < {BR // 16}; ii++) {{
        int row = warp + ii * 16;
#pragma unroll
        for (int jj = 0; jj < 4; jj++) {{
            int col = lane + jj * 32;
            z[row * 136 + col] = __float2bfloat16(pbuf[ii][jj]);
            du[row * 136 + col] = __float2bfloat16(lbuf[ii][jj]);
        }}
    }}
    __syncthreads();
    for (int p = tid; p < {BR} * 16; p += 512) {{
        int row = p / 16, col = (p % 16) * 8;
        if (t0 + row < {T}) {{
            long long dst =
                ((long long)h * {B * T} + batch * {T} + t0 + row) * 128 + col;
            *(int4 *)(out3_p + dst) = *(int4 *)(z + row * 136 + col);
            *(int4 *)(out4_p + dst) = *(int4 *)(du + row * 136 + col);
        }}
    }}
    for (int j = 0; j < {NR}; j++) {{
        wmma::fill_fragment(a0[j], 0.f);
        wmma::fill_fragment(a1[j], 0.f);
    }}
#pragma unroll
    for (int k = 0; k < 128; k += 16) {{
        cyfa_fast_load(bf, w + k * 136 + cw * 16, 136);
#pragma unroll
        for (int j = 0; j < {NR}; j++) {{
            cyfa_fast_load(af, z + (row_group * {BR // 2} + j * 16) * 136 + k, 136);
            wmma::mma_sync(a0[j], af, bf, a0[j]);
            cyfa_fast_load(af, du + (row_group * {BR // 2} + j * 16) * 136 + k, 136);
            wmma::mma_sync(a1[j], af, bf, a1[j]);
        }}
    }}
    __syncthreads();
    for (int j = 0; j < {NR}; j++) {{
        wmma::store_matrix_sync(score + (row_group * {BR // 2} + j * 16) * 136 + cw * 16, a0[j],
                                136, wmma::mem_row_major);
        wmma::store_matrix_sync(dp + (row_group * {BR // 2} + j * 16) * 136 + cw * 16, a1[j],
                                136, wmma::mem_row_major);
    }}
    __syncthreads();
    for (int p = tid; p < {BR} * 64; p += 512) {{
        int row = p / 64, m = (p % 64) * 2, t = t0 + row;
        if (t < {T}) {{
            long long q = ((long long)bh * {T} + t) * 128 + m,
                      ph = ((long long)bh * {T} + t) * 64 + m / 2;
            float re = (float)__float2bfloat16((float)in0_p[q]),
                  ro = (float)__float2bfloat16((float)in0_p[q + 1]);
            float dwe = (float)in4_p[q], dwo = (float)in4_p[q + 1], co = in1_p[ph], si = in2_p[ph];
            float ue = score[row * 136 + m], uo = score[row * 136 + m + 1], ge = dp[row * 136 + m],
                  go = dp[row * 136 + m + 1];
            __nv_bfloat162 gv = __floats2bfloat162_rn(co * ge + si * go, -si * ge + co * go);
            *(unsigned *)(out0_p + q) = *(unsigned *)&gv;
            out1_p[ph] = dwe * ue + dwo * uo + ge * re + go * ro;
            out2_p[ph] = dwe * uo - dwo * ue - ge * ro + go * re;
        }}
    }}
}}
cudaFuncSetAttribute(readout_adjoint, cudaFuncAttributeMaxDynamicSharedMemorySize, {smem});
readout_adjoint<<<dim3(({T} + {BR - 1}) / {BR}, {B * H}), 512,
                  {smem}>>>(@ARGS);
"""
    dr, dc, ds, p, dl, z, du = jt.code(
        [
            raw.shape,
            c.shape,
            s.shape,
            [H, B * T, M],
            [H, B * T, M],
            [H, B * T, M],
            [H, B * T, M],
        ],
        [
            "bfloat16",
            "float32",
            "float32",
            "bfloat16",
            "bfloat16",
            "bfloat16",
            "bfloat16",
        ],
        [raw, c, s, table, dw],
        cuda_header=WMMA_HEADER,
        cuda_src=src,
    )
    S = min(64, max(1, B * T // 256))
    pp = p.reshape((H, S, B * T // S, M))
    ll = dl.reshape(pp.shape)
    uu = du.reshape((H, S, B * T // S, M))
    zz = z.reshape(uu.shape)
    dt = (_mm(pp, uu, ta=True) + _mm(ll, zz, ta=True)).sum(1)
    return dr, dc, ds, dt[:, :R, :]


def _readout_grad_full(self, dw):
    return _readout_grad_wide(self, dw)


def _readout_grad_compact(self, dw):
    B, H, T, M = self.dims
    R = M - 1
    BR = 32
    NR = BR // 32
    raw, lam, table = self.raw, self.lam, self.table
    dw = dw.bfloat16()
    smem = 128 * 136 * 2 + 2 * BR * 136 * 4
    src = f"""

__global__ void readout_adjoint(@ARGS_DEF) {{
    @PRECALC
    using namespace nvcuda;
    int bh = blockIdx.y, h = bh % {H}, batch = bh / {H},
        t0 = blockIdx.x * {BR}, tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    int row_group = warp / 8, cw = warp % 8;
    extern __shared__ __align__(32) unsigned char smem[];
    __nv_bfloat16 *w = (__nv_bfloat16 *)smem, *z = w + 128 * 136, *du = z + {BR} * 136;
    float *score = (float *)z, *dp = score + {BR} * 136;
    for (int p = tid; p < 128 * 128; p += 512) {{
        int r = p / 128, m = p % 128;
        w[r * 136 + m] =
            __float2bfloat16(r < 127 ? in2_p[(long long)h * 127 * 128 + r * 128 + m] : 0.f);
    }}
    for (int p = tid; p < {BR} * 64; p += 512) {{
        int row = p / 64, m = (p % 64) * 2, t = t0 + row;
        float ze = 0, zo = 0, de = 0, do_ = 0;
        if (t < {T}) {{
            long long q = ((long long)bh * {T} + t) * 128 + m,
                      ph = ((long long)bh * {T} + t) * 64 + m / 2;
            float re = (float)__float2bfloat16((float)in0_p[q]),
                  ro = (float)__float2bfloat16((float)in0_p[q + 1]);
            float dwe = (float)in3_p[q], dwo = (float)in3_p[q + 1],
                  co = __cosf(cyfa_clock_phase(in1_p[(long long)bh * {T} + t], m / 2)),
                  si = __sinf(cyfa_clock_phase(in1_p[(long long)bh * {T} + t], m / 2));
            ze = co * re - si * ro;
            zo = si * re + co * ro;
            de = co * dwe - si * dwo;
            do_ = si * dwe + co * dwo;
            long long dst = ((long long)h * {B * T} + batch * {T} + t) * 128 + m;
            __nv_bfloat162 zv = __floats2bfloat162_rn(ze, zo);
            *(unsigned *)(out4_p + dst) = *(unsigned *)&zv;
            __nv_bfloat162 dv = __floats2bfloat162_rn(de, do_);
            *(unsigned *)(out5_p + dst) = *(unsigned *)&dv;
        }}
        z[row * 136 + m] = __float2bfloat16(ze);
        z[row * 136 + m + 1] = __float2bfloat16(zo);
        du[row * 136 + m] = __float2bfloat16(de);
        du[row * 136 + m + 1] = __float2bfloat16(do_);
    }}
    __syncthreads();
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bc;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> a0[{NR}], a1[{NR}];
    for (int j = 0; j < {NR}; j++) {{
        wmma::fill_fragment(a0[j], 0.f);
        wmma::fill_fragment(a1[j], 0.f);
    }}
#pragma unroll
    for (int k = 0; k < 128; k += 16) {{
        cyfa_fast_load(bc, w + cw * 16 * 136 + k, 136);
#pragma unroll
        for (int j = 0; j < {NR}; j++) {{
            cyfa_fast_load(af, z + (row_group * {BR // 2} + j * 16) * 136 + k, 136);
            wmma::mma_sync(a0[j], af, bc, a0[j]);
            cyfa_fast_load(af, du + (row_group * {BR // 2} + j * 16) * 136 + k, 136);
            wmma::mma_sync(a1[j], af, bc, a1[j]);
        }}
    }}
    __syncthreads();
    for (int j = 0; j < {NR}; j++) {{
        wmma::store_matrix_sync(score + (row_group * {BR // 2} + j * 16) * 136 + cw * 16, a0[j],
                                136, wmma::mem_row_major);
        wmma::store_matrix_sync(dp + (row_group * {BR // 2} + j * 16) * 136 + cw * 16, a1[j],
                                136, wmma::mem_row_major);
    }}
    __syncthreads();
    float pbuf[{BR // 16}][4], lbuf[{BR // 16}][4];
#pragma unroll
    for (int ii = 0; ii < {BR // 16}; ii++) {{
        int row = warp + ii * 16;
        float mx = -1.e30f;
#pragma unroll
        for (int jj = 0; jj < 4; jj++) {{
            int col = lane + jj * 32;
            if (col < 127)
                mx = fmaxf(mx, score[row * 136 + col]);
        }}
        for (int d = 16; d > 0; d /= 2)
            mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, d));
        mx = __shfl_sync(0xffffffff, mx, 0);
        float sum = 0.f, ex[4];
#pragma unroll
        for (int jj = 0; jj < 4; jj++) {{
            int col = lane + jj * 32;
            ex[jj] = col < 127 ? expf(score[row * 136 + col] - mx) : 0.f;
            sum += ex[jj];
        }}
        for (int d = 16; d > 0; d /= 2)
            sum += __shfl_down_sync(0xffffffff, sum, d);
        sum = __shfl_sync(0xffffffff, sum, 0);
        float correction = 0.f;
#pragma unroll
        for (int jj = 0; jj < 4; jj++) {{
            int col = lane + jj * 32;
            pbuf[ii][jj] = ex[jj] / sum;
            correction += pbuf[ii][jj] * dp[row * 136 + col];
        }}
        for (int d = 16; d > 0; d /= 2)
            correction += __shfl_down_sync(0xffffffff, correction, d);
        correction = __shfl_sync(0xffffffff, correction, 0);
#pragma unroll
        for (int jj = 0; jj < 4; jj++) {{
            int col = lane + jj * 32;
            lbuf[ii][jj] = pbuf[ii][jj] * (dp[row * 136 + col] - correction);
        }}
    }}
    __syncthreads();
#pragma unroll
    for (int ii = 0; ii < {BR // 16}; ii++) {{
        int row = warp + ii * 16;
#pragma unroll
        for (int jj = 0; jj < 4; jj++) {{
            int col = lane + jj * 32;
            z[row * 136 + col] = __float2bfloat16(pbuf[ii][jj]);
            du[row * 136 + col] = __float2bfloat16(lbuf[ii][jj]);
        }}
    }}
    __syncthreads();
    for (int p = tid; p < {BR} * 16; p += 512) {{
        int row = p / 16, col = (p % 16) * 8;
        if (t0 + row < {T}) {{
            long long dst =
                ((long long)h * {B * T} + batch * {T} + t0 + row) * 128 + col;
            *(int4 *)(out2_p + dst) = *(int4 *)(z + row * 136 + col);
            *(int4 *)(out3_p + dst) = *(int4 *)(du + row * 136 + col);
        }}
    }}
    for (int j = 0; j < {NR}; j++) {{
        wmma::fill_fragment(a0[j], 0.f);
        wmma::fill_fragment(a1[j], 0.f);
    }}
#pragma unroll
    for (int k = 0; k < 128; k += 16) {{
        cyfa_fast_load(bf, w + k * 136 + cw * 16, 136);
#pragma unroll
        for (int j = 0; j < {NR}; j++) {{
            cyfa_fast_load(af, z + (row_group * {BR // 2} + j * 16) * 136 + k, 136);
            wmma::mma_sync(a0[j], af, bf, a0[j]);
            cyfa_fast_load(af, du + (row_group * {BR // 2} + j * 16) * 136 + k, 136);
            wmma::mma_sync(a1[j], af, bf, a1[j]);
        }}
    }}
    __syncthreads();
    for (int j = 0; j < {NR}; j++) {{
        wmma::store_matrix_sync(score + (row_group * {BR // 2} + j * 16) * 136 + cw * 16, a0[j],
                                136, wmma::mem_row_major);
        wmma::store_matrix_sync(dp + (row_group * {BR // 2} + j * 16) * 136 + cw * 16, a1[j],
                                136, wmma::mem_row_major);
    }}
    __syncthreads();
    for (int p = tid; p < {BR} * 64; p += 512) {{
        int row = p / 64, m = (p % 64) * 2, t = t0 + row;
        if (t < {T}) {{
            long long q = ((long long)bh * {T} + t) * 128 + m,
                      ph = ((long long)bh * {T} + t) * 64 + m / 2;
            float re = (float)__float2bfloat16((float)in0_p[q]),
                  ro = (float)__float2bfloat16((float)in0_p[q + 1]);
            float dwe = (float)in3_p[q], dwo = (float)in3_p[q + 1],
                  co = __cosf(cyfa_clock_phase(in1_p[(long long)bh * {T} + t], m / 2)),
                  si = __sinf(cyfa_clock_phase(in1_p[(long long)bh * {T} + t], m / 2));
            float ue = score[row * 136 + m], uo = score[row * 136 + m + 1], ge = dp[row * 136 + m],
                  go = dp[row * 136 + m + 1];
            __nv_bfloat162 gv = __floats2bfloat162_rn(co * ge + si * go, -si * ge + co * go);
            *(unsigned *)(out0_p + q) = *(unsigned *)&gv;
            float dc = dwe * ue + dwo * uo + ge * re + go * ro,
                  ds = dwe * uo - dwo * ue - ge * ro + go * re;
            ((float *)w)[row * 64 + m / 2] =
                (dc * (-si) + ds * co) * ((m / 2) * 0.04947390005653217f);
        }}
    }}
    __syncthreads();
    for (int ii = 0; ii < {BR // 16}; ii++) {{
        int row = warp + ii * 16, t = t0 + row;
        float force = ((float *)w)[row * 64 + lane] + ((float *)w)[row * 64 + lane + 32];
        for (int d = 16; d > 0; d /= 2)
            force += __shfl_down_sync(0xffffffff, force, d);
        if (lane == 0 && t < {T})
            out1_p[(long long)bh * {T} + t] = force;
    }}
}}
cudaFuncSetAttribute(readout_adjoint, cudaFuncAttributeMaxDynamicSharedMemorySize, {smem});
readout_adjoint<<<dim3(({T} + {BR - 1}) / {BR}, {B * H}), 512,
                  {smem}>>>(@ARGS);
"""
    dr, dlam, p, dl, z, du = jt.code(
        [
            raw.shape,
            lam.shape,
            [H, B * T, M],
            [H, B * T, M],
            [H, B * T, M],
            [H, B * T, M],
        ],
        ["bfloat16", "float32", "bfloat16", "bfloat16", "bfloat16", "bfloat16"],
        [raw, lam, table, dw],
        cuda_header=WMMA_HEADER + CLOCK_PHASE_HEADER,
        cuda_src=src,
    )
    S = min(64, max(1, B * T // 256))
    pp = p.reshape((H, S, B * T // S, M))
    ll = dl.reshape(pp.shape)
    uu = du.reshape((H, S, B * T // S, M))
    zz = z.reshape(uu.shape)
    dt = (_mm(pp, uu, ta=True) + _mm(ll, zz, ta=True)).sum(1)
    return dr, dlam, dt[:, :R, :]


def readout_grad_compact(self, dw):
    return _readout_grad_compact(self, dw)


def _readout_forward_wide(q, k, v, g, states, c, s, table, scale):
    B, H, T, K = q.shape
    BR = 64
    NR = 2
    compact_raw = K <= 256
    tail = f"""

// Reuse the state tile storage for the learned table and the query tile for rotations.
for (int p = tid; p < 128 * 16; p += 512) {{
    int r = p / 16, m = (p % 16) * 8;
    int4 data = make_int4(0, 0, 0, 0);
    if (r < 127)
        data = *(const int4 *)((const __nv_bfloat16 *)in7_p +
                               (long long)(bh % {H}) * 127 * 128 + r * 128 + m);
    *(int4 *)(buf + r * 136 + m) = data;
}}
unsigned packed_rotation[({BR} * 64 + 511) / 512];
#pragma unroll
for (int ii = 0; ii < ({BR} * 64 + 511) / 512; ii++) {{
    int p = tid + ii * 512;
    int row = p / 64, m = (p % 64) * 2;
    long long ph = ((long long)bh * {T} + t0 + row) * 64 + m / 2;
    float e = work[row * 136 + m], o = work[row * 136 + m + 1], co = in5_p[ph], si = in6_p[ph];
    __nv_bfloat162 pack = __floats2bfloat162_rn(co * e - si * o, si * e + co * o);
    packed_rotation[ii] = *(unsigned *)&pack;
}}
__syncthreads();
#pragma unroll
for (int ii = 0; ii < ({BR} * 64 + 511) / 512; ii++) {{
    int p = tid + ii * 512;
    int row = p / 64, m = (p % 64) * 2;
    *(unsigned *)(qs + row * 136 + m) = packed_rotation[ii];
}}
__syncthreads();
for (int r = 0; r < {NR}; r++)
    wmma::fill_fragment(acc[r], 0.f);
#pragma unroll
for (int ss = 0; ss < 128; ss += 16) {{
    cyfa_fast_load(bc, buf + warp * 16 * 136 + ss, 136);
#pragma unroll
    for (int r = 0; r < {NR}; r++) {{
        cyfa_fast_load(af, qs + (row_group * 32 + r * 16) * 136 + ss, 136);
        wmma::mma_sync(acc[r], af, bc, acc[r]);
    }}
}}
__syncthreads();
for (int r = 0; r < {NR}; r++)
    wmma::store_matrix_sync(work + (row_group * 32 + r * 16) * 136 + warp * 16, acc[r], 136,
                            wmma::mem_row_major);
__syncthreads();
float pbuf[{BR // 16}][4];
#pragma unroll
for (int ii = 0; ii < {BR // 16}; ii++) {{
    int row = warp + row_group * 8 + ii * 16;
    float mx = -1.e30f;
#pragma unroll
    for (int jj = 0; jj < 4; jj++) {{
        int col = tid % 32 + jj * 32;
        if (col < 127)
            mx = fmaxf(mx, work[row * 136 + col]);
    }}
    for (int d = 16; d > 0; d /= 2)
        mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, d));
    mx = __shfl_sync(0xffffffff, mx, 0);
    float sum = 0.f;
#pragma unroll
    for (int jj = 0; jj < 4; jj++) {{
        int col = tid % 32 + jj * 32;
        pbuf[ii][jj] = col < 127 ? expf(work[row * 136 + col] - mx) : 0.f;
        sum += pbuf[ii][jj];
    }}
    for (int d = 16; d > 0; d /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, d);
    sum = __shfl_sync(0xffffffff, sum, 0);
#pragma unroll
    for (int jj = 0; jj < 4; jj++)
        pbuf[ii][jj] /= sum;
}}
__syncthreads();
#pragma unroll
for (int ii = 0; ii < {BR // 16}; ii++) {{
    int row = warp + row_group * 8 + ii * 16;
#pragma unroll
    for (int jj = 0; jj < 4; jj++) {{
        int col = tid % 32 + jj * 32;
        qs[row * 136 + col] = __float2bfloat16(pbuf[ii][jj]);
    }}
}}
__syncthreads();
for (int r = 0; r < {NR}; r++)
    wmma::fill_fragment(acc[r], 0.f);
#pragma unroll
for (int ss = 0; ss < 128; ss += 16) {{
    cyfa_fast_load(bf, buf + ss * 136 + warp * 16, 136);
#pragma unroll
    for (int r = 0; r < {NR}; r++) {{
        cyfa_fast_load(af, qs + (row_group * 32 + r * 16) * 136 + ss, 136);
        wmma::mma_sync(acc[r], af, bf, acc[r]);
    }}
}}
__syncthreads();
for (int r = 0; r < {NR}; r++)
    wmma::store_matrix_sync(work + (row_group * 32 + r * 16) * 136 + warp * 16, acc[r], 136,
                            wmma::mem_row_major);
__syncthreads();
for (int p = tid; p < {BR} * 64; p += 512) {{
    int row = p / 64, m = (p % 64) * 2;
    long long ph = ((long long)bh * {T} + t0 + row) * 64 + m / 2,
              pos = ((long long)bh * {T} + t0 + row) * 128 + m;
    float e = work[row * 136 + m], o = work[row * 136 + m + 1], co = in5_p[ph], si = in6_p[ph];
    __nv_bfloat162 value = __floats2bfloat162_rn(co * e + si * o, -si * e + co * o);
    *(unsigned *)(out1_p + pos) = *(unsigned *)&value;
}}
"""
    src, header = _output_wide(
        q,
        k,
        v,
        g,
        states,
        scale,
        True,
        return_source=True,
        _rows=BR,
        epilogue=tail,
        compact_raw=compact_raw,
        kernel_name="first_readout",
    )
    raw, weights = jt.code(
        [[B, H, T, 128], [B, H, T, 128]],
        ["bfloat16" if compact_raw else "float32", "bfloat16"],
        [q, k, v, g, states, c, s, table.bfloat16()],
        cuda_header=header,
        cuda_src=src,
    )
    return raw, weights


class _FirstReadoutWide(jt.Function):
    def execute(self, q, k, write, g, initial, c, s, table, scale):
        B, H, T, K = q.shape
        self.inputs = q, k, write, g, initial
        self.scale = scale
        self.raw_output = True
        self.gc = local_log2_cumsum(g)
        self.states, final = state_forward(k, write, self.gc, initial)
        self.raw, weights = _readout_forward_wide(
            q, k, write, self.gc, self.states, c, s, table, scale
        )
        self.c, self.s, self.table, self.dims = c, s, table, (B, H, T, 128)
        return weights, final

    def grad(self, dw, df):
        if dw is None:
            dr = jt.zeros(self.raw.shape, dtype="bfloat16")
            dc = jt.zeros_like(self.c)
            ds = jt.zeros_like(self.s)
            dt = jt.zeros_like(self.table)
        else:
            dr, dc, ds, dt = _readout_grad_full(self, dw)
        dq, dk, dwrite, dg, dh0, _, _ = scalar_grad(self, dr, df)
        return dq, dk, dwrite, dg, dh0, dc, ds, dt, None


def _first_readout_wide(q, k, write, g, initial, c, s, table, scale):
    return _FirstReadoutWide.apply(q, k, write, g, initial, c, s, table, scale)


def _readout_forward_compact(q, k, v, g, states, lam, table, scale, q_norm=None):
    B, H, T, K = q.shape
    BR = 64
    NR = 4
    compact_raw = K <= 256
    tail = f"""

// Reuse the state tile storage for the learned table and the query tile for rotations.
for (int p = tid; p < 128 * 16; p += 256) {{
    int r = p / 16, m = (p % 16) * 8;
    int4 data = make_int4(0, 0, 0, 0);
    if (r < 127)
        data = *(const int4 *)((const __nv_bfloat16 *)in6_p +
                               (long long)(bh % {H}) * 127 * 128 + r * 128 + m);
    *(int4 *)(buf + r * 136 + m) = data;
}}
unsigned packed_rotation[({BR} * 64 + 255) / 256];
#pragma unroll
for (int ii = 0; ii < ({BR} * 64 + 255) / 256; ii++) {{
    int p = tid + ii * 256;
    int row = p / 64, m = (p % 64) * 2;
    long long ph = ((long long)bh * {T} + t0 + row) * 64 + m / 2;
    float e = work[row * 136 + m], o = work[row * 136 + m + 1],
          co = __cosf(cyfa_clock_phase(in5_p[(long long)bh * {T} + t0 + row], m / 2)),
          si = __sinf(cyfa_clock_phase(in5_p[(long long)bh * {T} + t0 + row], m / 2));
    __nv_bfloat162 pack = __floats2bfloat162_rn(co * e - si * o, si * e + co * o);
    packed_rotation[ii] = *(unsigned *)&pack;
}}
__syncthreads();
#pragma unroll
for (int ii = 0; ii < ({BR} * 64 + 255) / 256; ii++) {{
    int p = tid + ii * 256;
    int row = p / 64, m = (p % 64) * 2;
    *(unsigned *)(qs + row * 136 + m) = packed_rotation[ii];
}}
__syncthreads();
for (int r = 0; r < {NR}; r++)
    wmma::fill_fragment(acc[r], 0.f);
#pragma unroll
for (int ss = 0; ss < 128; ss += 16) {{
    cyfa_fast_load(bc, buf + warp * 16 * 136 + ss, 136);
#pragma unroll
    for (int r = 0; r < {NR}; r++) {{
        cyfa_fast_load(af, qs + (row_group * 32 + r * 16) * 136 + ss, 136);
        wmma::mma_sync(acc[r], af, bc, acc[r]);
    }}
}}
__syncthreads();
for (int r = 0; r < {NR}; r++)
    wmma::store_matrix_sync(work + (row_group * 32 + r * 16) * 136 + warp * 16, acc[r], 136,
                            wmma::mem_row_major);
__syncthreads();
float pbuf[{BR // 8}][4];
#pragma unroll
for (int ii = 0; ii < {BR // 8}; ii++) {{
    int row = warp + ii * 8;
    float mx = -1.e30f;
#pragma unroll
    for (int jj = 0; jj < 4; jj++) {{
        int col = tid % 32 + jj * 32;
        if (col < 127)
            mx = fmaxf(mx, work[row * 136 + col]);
    }}
    for (int d = 16; d > 0; d /= 2)
        mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, d));
    mx = __shfl_sync(0xffffffff, mx, 0);
    float sum = 0.f;
#pragma unroll
    for (int jj = 0; jj < 4; jj++) {{
        int col = tid % 32 + jj * 32;
        pbuf[ii][jj] = col < 127 ? expf(work[row * 136 + col] - mx) : 0.f;
        sum += pbuf[ii][jj];
    }}
    for (int d = 16; d > 0; d /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, d);
    sum = __shfl_sync(0xffffffff, sum, 0);
#pragma unroll
    for (int jj = 0; jj < 4; jj++)
        pbuf[ii][jj] /= sum;
}}
__syncthreads();
#pragma unroll
for (int ii = 0; ii < {BR // 8}; ii++) {{
    int row = warp + ii * 8;
#pragma unroll
    for (int jj = 0; jj < 4; jj++) {{
        int col = tid % 32 + jj * 32;
        qs[row * 136 + col] = __float2bfloat16(pbuf[ii][jj]);
    }}
}}
__syncthreads();
for (int r = 0; r < {NR}; r++)
    wmma::fill_fragment(acc[r], 0.f);
#pragma unroll
for (int ss = 0; ss < 128; ss += 16) {{
    cyfa_fast_load(bf, buf + ss * 136 + warp * 16, 136);
#pragma unroll
    for (int r = 0; r < {NR}; r++) {{
        cyfa_fast_load(af, qs + (row_group * 32 + r * 16) * 136 + ss, 136);
        wmma::mma_sync(acc[r], af, bf, acc[r]);
    }}
}}
__syncthreads();
for (int r = 0; r < {NR}; r++)
    wmma::store_matrix_sync(work + (row_group * 32 + r * 16) * 136 + warp * 16, acc[r], 136,
                            wmma::mem_row_major);
__syncthreads();
for (int p = tid; p < {BR} * 64; p += 256) {{
    int row = p / 64, m = (p % 64) * 2;
    long long ph = ((long long)bh * {T} + t0 + row) * 64 + m / 2,
              pos = ((long long)bh * {T} + t0 + row) * 128 + m;
    float e = work[row * 136 + m], o = work[row * 136 + m + 1],
          co = __cosf(cyfa_clock_phase(in5_p[(long long)bh * {T} + t0 + row], m / 2)),
          si = __sinf(cyfa_clock_phase(in5_p[(long long)bh * {T} + t0 + row], m / 2));
    __nv_bfloat162 value = __floats2bfloat162_rn(co * e + si * o, -si * e + co * o);
    *(unsigned *)(out1_p + pos) = *(unsigned *)&value;
}}
"""
    new_q = None
    kernel_name = "first_readout_w256"
    norm_inputs = []
    if q_norm is not None:
        assert jt.flags.no_grad and K == 256, (
            "fused query normalization is inference-only"
        )
        qw, qrs = q_norm
        new_q = f"""
for (int p = tid; p < {BR} * 16; p += 256) {{
    int r = p / 16, c = (p % 16) * 8;
    long long qp =
        (((long long)(bh / {H}) * {T} + t0 + r) * {H} + bh % {H}) *
            {K} +
        base + c;
    __align__(16) __nv_bfloat16 data[8];
    *(int4 *)data = *(const int4 *)(in0_p + qp);
    float rs = in8_p[(long long)bh * {T} + t0 + r];
#pragma unroll
    for (int j = 0; j < 8; j++)
        data[j] = __float2bfloat16((float)data[j] * rs * in7_p[base + c + j]);
    *(int4 *)(qs + r * 136 + c) = *(int4 *)data;
}}"""
        kernel_name = "first_readout_qnorm_w256"
        norm_inputs = [qw, qrs]
    src, header = _output_wide(
        q,
        k,
        v,
        g,
        states,
        scale,
        True,
        return_source=True,
        _rows=BR,
        _threads=256,
        epilogue=tail,
        compact_raw=compact_raw,
        query_load=new_q,
        kernel_name=kernel_name,
    )
    raw, weights = jt.code(
        [[B, H, T, 128], [B, H, T, 128]],
        ["bfloat16" if compact_raw else "float32", "bfloat16"],
        [q, k, v, g, states, lam, table.bfloat16()] + norm_inputs,
        cuda_header=header + CLOCK_PHASE_HEADER,
        cuda_src=src,
    )
    return raw, weights


class FirstReadoutCompact(jt.Function):
    def execute(self, q, k, write, g, initial, lam, table, scale, q_norm):
        self.q_norm = q_norm
        B, H, T, K = q.shape
        self.inputs = q, k, write, g, initial
        self.scale = scale
        self.raw_output = True
        self.gc = local_log2_cumsum(g)
        self.states, final = state_forward(k, write, self.gc, initial)
        self.raw, weights = _readout_forward_compact(
            q, k, write, self.gc, self.states, lam, table, scale, q_norm
        )
        self.lam, self.table, self.dims = lam, table, (B, H, T, 128)
        return weights, final

    def grad(self, dw, df):
        assert self.q_norm is None, (
            "inference-only fused query norm cannot be differentiated"
        )
        if dw is None:
            dr = jt.zeros(self.raw.shape, dtype="bfloat16")
            dlam = jt.zeros_like(self.lam)
            dt = jt.zeros_like(self.table)
        else:
            dr, dlam, dt = readout_grad_compact(self, dw)
        dq, dk, dwrite, dg, dh0, _, _ = scalar_grad(self, dr, df)
        return dq, dk, dwrite, dg, dh0, dlam, dt, None, None


def first_readout_compact(q, k, write, g, initial, lam, table, scale, q_norm=None):
    return FirstReadoutCompact.apply(q, k, write, g, initial, lam, table, scale, q_norm)


def _readout_forward_small(q, k, v, g, states, c, s, table, scale):
    B, H, T, K = q.shape
    BR = 32
    NR = BR // 16
    tail = f"""

// Reuse the state tile storage for the learned table and the query tile for rotations.
for (int p = tid; p < 128 * 16; p += 256) {{
    int r = p / 16, m = (p % 16) * 8;
    int4 data = make_int4(0, 0, 0, 0);
    if (r < 127)
        data = *(const int4 *)((const __nv_bfloat16 *)in7_p +
                               (long long)(bh % {H}) * 127 * 128 + r * 128 + m);
    *(int4 *)(buf + r * 136 + m) = data;
}}
for (int p = tid; p < {BR} * 64; p += 256) {{
    int row = p / 64, m = (p % 64) * 2;
    long long ph = ((long long)bh * {T} + t0 + row) * 64 + m / 2;
    float e = work[row * 136 + m], o = work[row * 136 + m + 1], co = in5_p[ph], si = in6_p[ph];
    qs[row * 136 + m] = __float2bfloat16(co * e - si * o);
    qs[row * 136 + m + 1] = __float2bfloat16(si * e + co * o);
}}
__syncthreads();
for (int r = 0; r < {NR}; r++)
    wmma::fill_fragment(acc[r], 0.f);
#pragma unroll
for (int ss = 0; ss < 128; ss += 16) {{
    cyfa_fast_load(bc, buf + warp * 16 * 136 + ss, 136);
#pragma unroll
    for (int r = 0; r < {NR}; r++) {{
        cyfa_fast_load(af, qs + r * 16 * 136 + ss, 136);
        wmma::mma_sync(acc[r], af, bc, acc[r]);
    }}
}}
for (int r = 0; r < {NR}; r++)
    wmma::store_matrix_sync(work + r * 16 * 136 + warp * 16, acc[r], 136, wmma::mem_row_major);
__syncthreads();
for (int row = warp; row < {BR}; row += 8) {{
    float mx = -1.e30f;
    for (int col = tid % 32; col < 127; col += 32)
        mx = fmaxf(mx, work[row * 136 + col]);
    for (int d = 16; d > 0; d /= 2)
        mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, d));
    mx = __shfl_sync(0xffffffff, mx, 0);
    float sum = 0.f;
    for (int col = tid % 32; col < 128; col += 32) {{
        float p = col < 127 ? expf(work[row * 136 + col] - mx) : 0.f;
        work[row * 136 + col] = p;
        sum += p;
    }}
    for (int d = 16; d > 0; d /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, d);
    sum = __shfl_sync(0xffffffff, sum, 0);
    for (int col = tid % 32; col < 128; col += 32)
        qs[row * 136 + col] = __float2bfloat16(work[row * 136 + col] / sum);
}}
__syncthreads();
for (int r = 0; r < {NR}; r++)
    wmma::fill_fragment(acc[r], 0.f);
#pragma unroll
for (int ss = 0; ss < 128; ss += 16) {{
    cyfa_fast_load(bf, buf + ss * 136 + warp * 16, 136);
#pragma unroll
    for (int r = 0; r < {NR}; r++) {{
        cyfa_fast_load(af, qs + r * 16 * 136 + ss, 136);
        wmma::mma_sync(acc[r], af, bf, acc[r]);
    }}
}}
for (int r = 0; r < {NR}; r++)
    wmma::store_matrix_sync(work + r * 16 * 136 + warp * 16, acc[r], 136, wmma::mem_row_major);
__syncthreads();
for (int p = tid; p < {BR} * 64; p += 256) {{
    int row = p / 64, m = (p % 64) * 2;
    long long ph = ((long long)bh * {T} + t0 + row) * 64 + m / 2,
              pos = ((long long)bh * {T} + t0 + row) * 128 + m;
    float e = work[row * 136 + m], o = work[row * 136 + m + 1], co = in5_p[ph], si = in6_p[ph];
    __nv_bfloat162 value = __floats2bfloat162_rn(co * e + si * o, -si * e + co * o);
    *(unsigned *)(out1_p + pos) = *(unsigned *)&value;
}}
"""
    src, header = output_forward(
        q,
        k,
        v,
        g,
        states,
        scale,
        True,
        return_source=True,
        _rows=BR,
        epilogue=tail,
        compact_raw=False,
        kernel_name="first_readout",
    )
    raw, weights = jt.code(
        [[B, H, T, 128], [B, H, T, 128]],
        ["float32", "bfloat16"],
        [q, k, v, g, states, c, s, table.bfloat16()],
        cuda_header=header,
        cuda_src=src,
    )
    return raw, weights


class FirstReadout(jt.Function):
    def execute(self, q, k, write, g, initial, c, s, table, scale):
        B, H, T, K = q.shape
        self.inputs = q, k, write, g, initial
        self.scale = scale
        self.raw_output = True
        self.gc = local_log2_cumsum(g)
        self.states, final = state_forward(k, write, self.gc, initial)
        self.raw, weights = _readout_forward_small(
            q, k, write, self.gc, self.states, c, s, table, scale
        )
        self.c, self.s, self.table, self.dims = c, s, table, (B, H, T, 128)
        return weights, final

    def grad(self, dw, df):
        if dw is None:
            dr = jt.zeros(self.raw.shape, dtype="bfloat16")
            dc = jt.zeros_like(self.c)
            ds = jt.zeros_like(self.s)
            dt = jt.zeros_like(self.table)
        else:
            dr, dc, ds, dt = _readout_grad_small(self, dw)
        dq, dk, dwrite, dg, dh0, _, _ = scalar_grad(self, dr, df)
        return dq, dk, dwrite, dg, dh0, dc, ds, dt, None


def first_readout(q, k, write, g, initial, c, s, table, scale, q_norm=None):
    if s is None:
        from .utils import use_large

        if not use_large(q):
            raise ValueError("Compact phase protocol requires a large fused input")
        return first_readout_compact(q, k, write, g, initial, c, table, scale, q_norm)
    assert q_norm is None, "query norm fusion requires compact prefill"
    from .utils import use_large

    if use_large(q):
        return _first_readout_wide(q, k, write, g, initial, c, s, table, scale)
    return FirstReadout.apply(q, k, write, g, initial, c, s, table, scale)
