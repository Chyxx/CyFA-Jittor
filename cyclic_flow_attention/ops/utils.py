# SPDX-License-Identifier: MIT
# See LICENSE, LICENSE.fla and THIRD_PARTY_NOTICES.txt.
"""Clock, phase, layout and matrix primitives shared by the CyFA kernels."""

import math

import jittor as jt

WMMA_HEADER = """

#include <mma.h>
#include <cuda_bf16.h>
using CyfaAR = nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                                      nvcuda::wmma::row_major>;
using CyfaAC = nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                                      nvcuda::wmma::col_major>;
using CyfaBR = nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                                      nvcuda::wmma::row_major>;
using CyfaBC = nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                                      nvcuda::wmma::col_major>;
template <bool Trans, bool Perm, class Frag>
__device__ __forceinline__ void cyfa_ld(Frag &f, const __nv_bfloat16 *p, int ld) {
    int lane = threadIdx.x % 32;
    unsigned addr = __cvta_generic_to_shared(p + (lane % 16) * ld + (lane / 16) * 8);
    unsigned r0, r1, r2, r3;
    if (Trans)
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                     : "r"(addr));
    else
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                     : "r"(addr));
    unsigned *out = (unsigned *)f.x;
    out[0] = r0;
    out[1] = Perm ? r2 : r1;
    out[2] = Perm ? r1 : r2;
    out[3] = r3;
}
__device__ __forceinline__ void cyfa_fast_load(CyfaAR &f, const __nv_bfloat16 *p, int ld) {
    cyfa_ld<false, false>(f, p, ld);
}
__device__ __forceinline__ void cyfa_fast_load(CyfaAC &f, const __nv_bfloat16 *p, int ld) {
    cyfa_ld<true, true>(f, p, ld);
}
__device__ __forceinline__ void cyfa_fast_load(CyfaBR &f, const __nv_bfloat16 *p, int ld) {
    cyfa_ld<true, false>(f, p, ld);
}
__device__ __forceinline__ void cyfa_fast_load(CyfaBC &f, const __nv_bfloat16 *p, int ld) {
    cyfa_ld<false, true>(f, p, ld);
}
"""

LARGE_MIN_TOKENS = 32768


def use_large(x):
    return x.shape[0] * x.shape[1] * x.shape[2] >= LARGE_MIN_TOKENS


PREFETCH_MIN_LENGTH = 1024


def use_prefetch(x):
    return x.shape[0] * x.shape[1] < 32 and x.shape[2] >= PREFETCH_MIN_LENGTH


_HEADER = """

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <dlfcn.h>
#include <stdexcept>
struct CyfaBlas {
    cublasHandle_t handle;
    using Gemm = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int,
                                    int, const void *, const void *, cudaDataType_t, int, long long,
                                    const void *, cudaDataType_t, int, long long, const void *,
                                    void *, cudaDataType_t, int, long long, int,
                                    cublasComputeType_t, cublasGemmAlgo_t);
    Gemm gemm;
    CyfaBlas() {
        void *lib = dlopen("libcublas.so.12", RTLD_NOW | RTLD_LOCAL);
        if (!lib)
            throw std::runtime_error(dlerror());
        auto create = (decltype(&cublasCreate_v2))dlsym(lib, "cublasCreate_v2");
        gemm = (decltype(gemm))dlsym(lib, "cublasGemmStridedBatchedEx");
        if (!create || !gemm || create(&handle) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("CyFA: cannot create cuBLAS handle");
    }
};
"""

_TYPES = {"float32": "CUDA_R_32F", "bfloat16": "CUDA_R_16BF", "float16": "CUDA_R_16F"}


def _mm(a, b, ta=False, tb=False, dtype="float32", compute="fp32"):
    if tuple(a.shape[:-2]) != tuple(b.shape[:-2]):
        raise ValueError("batched matmul needs equal batch shapes")
    m, k = (a.shape[-1], a.shape[-2]) if ta else a.shape[-2:]
    kb, n = (b.shape[-1], b.shape[-2]) if tb else b.shape[-2:]
    if k != kb:
        raise ValueError((a.shape, b.shape, ta, tb))
    batch = math.prod(a.shape[:-2])
    src = f"""
    static CyfaBlas blas;
    float alpha=1.f, beta=0.f;
    auto status = blas.gemm(blas.handle, CUBLAS_OP_{"T" if tb else "N"}, CUBLAS_OP_{"T" if ta else "N"},
        {n}, {m}, {k}, &alpha,
        in1_p, {_TYPES[str(b.dtype)]}, {b.shape[-1]}, (long long){b.shape[-2] * b.shape[-1]},
        in0_p, {_TYPES[str(a.dtype)]}, {a.shape[-1]}, (long long){a.shape[-2] * a.shape[-1]},
        &beta, out_p, {_TYPES[dtype]}, {n}, (long long){m * n}, {batch},
        {"CUBLAS_COMPUTE_32F_FAST_TF32" if compute == "tf32" else "CUBLAS_COMPUTE_32F"}, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("CyFA cuBLAS status " + std::to_string(status));
    """
    return jt.code(
        list(a.shape[:-2]) + [m, n], dtype, [a, b], cuda_header=_HEADER, cuda_src=src
    )


class _Bmm(jt.Function):
    def execute(self, a, b, ta, tb, dtype):
        self.a, self.b, self.ta, self.tb = a, b, ta, tb
        return _mm(a, b, ta, tb, dtype)

    def grad(self, dc):
        a, b, ta, tb = self.a, self.b, self.ta, self.tb
        # Preserve the input arithmetic dtype; all inner products accumulate in FP32.
        if ta:
            da = _mm(b, dc.cast(b.dtype), tb, True, "float32")
        else:
            da = _mm(dc.cast(b.dtype), b, False, not tb, "float32")
        if tb:
            db = _mm(dc.cast(a.dtype), a, True, ta, "float32")
        else:
            db = _mm(a, dc.cast(a.dtype), not ta, False, "float32")
        return da, db, None, None, None


def bmm(a, b, ta=False, tb=False, dtype="float32"):
    return _Bmm.apply(a, b, ta, tb, dtype)


class AffineScan(jt.Function):
    def execute(self, updates, decays, initial, dtype):
        # updates [B,H,N,K,V], decays [B,H,N], initial [B,H,K,V]
        self.u, self.a, self.initial = updates, decays, initial
        B, H, N, K, V = updates.shape
        self.dims = B, H, N, K, V
        src = f"""

__global__ void scan(@ARGS_DEF) {{
    @PRECALC
    int bh = blockIdx.y;
    int kv = blockIdx.x * blockDim.x + threadIdx.x;
    if (kv >= {K * V})
        return;
    float h = in2_p[(long long)bh * {K * V} + kv];
    for (int n = 0; n < {N}; n++) {{
        long long p = ((long long)bh * {N} + n) * {K * V} + kv;
        out0_p[p] = (out0_type)h;
        out2_p[p] = h;
        h = h * in1_p[bh * {N} + n] + in0_p[p];
    }}
    out1_p[(long long)bh * {K * V} + kv] = h;
}}
scan<<<dim3(({K * V} + 255) / 256, {B * H}), 256>>>(@ARGS);
"""
        self.states, self.final, self.states_fp32 = jt.code(
            [updates.shape, initial.shape, updates.shape],
            [dtype, "float32", "float32"],
            [updates, decays, initial],
            cuda_src=src,
        )
        return self.states, self.final

    def grad(self, ds, df):
        B, H, N, K, V = self.dims
        if ds is None:
            ds = jt.zeros_like(self.states)
        if df is None:
            df = jt.zeros_like(self.initial)
        src = f"""

__global__ void scan_grad(@ARGS_DEF) {{
    @PRECALC
    int bh = blockIdx.y;
    int kv = blockIdx.x * blockDim.x + threadIdx.x;
    if (kv >= {K * V})
        return;
    __shared__ float partial[8];
    float dh = in3_p[(long long)bh * {K * V} + kv];
    for (int n = {N} - 1; n >= 0; n--) {{
        long long p = ((long long)bh * {N} + n) * {K * V} + kv;
        out0_p[p] = dh;
        float da = dh * in0_p[p];
        for (int d = 16; d > 0; d /= 2)
            da += __shfl_down_sync(0xffffffff, da, d);
        if (threadIdx.x % 32 == 0)
            partial[threadIdx.x / 32] = da;
        __syncthreads();
        if (threadIdx.x == 0) {{
            float sum = 0.f;
            for (int w = 0; w < 8; w++)
                sum += partial[w];
            out1_p[((long long)bh * {N} + n) * {(K * V + 255) // 256} + blockIdx.x] = sum;
        }}
        __syncthreads();
        dh = dh * in1_p[bh * {N} + n] + (float)in2_p[p];
    }}
    out2_p[(long long)bh * {K * V} + kv] = dh;
}}
scan_grad<<<dim3(({K * V} + 255) / 256, {B * H}), 256>>>(@ARGS);
"""
        du, da, di = jt.code(
            [self.u.shape, [B, H, N, (K * V + 255) // 256], self.initial.shape],
            ["float32"] * 3,
            [self.states_fp32, self.a, ds, df],
            cuda_src=src,
        )
        return du, da.sum(-1), di, None


def affine_scan(updates, decays, initial, dtype="bfloat16"):
    return AffineScan.apply(updates, decays, initial, dtype)


class ClockScan(jt.Function):
    def execute(self, x):
        B, H, T = x.shape
        src = f"""

__global__ void clock_scan(@ARGS_DEF) {{
    @PRECALC
    int bh = blockIdx.x, tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    __shared__ float sums[8];
    float total = 0.f;
    for (int base = 0; base < {T}; base += 256) {{
        int ix = base + tid, iy = ix + 128;
        float x = ix < {T} ? in0_p[bh * {T} + ix] : 0.f,
              y = iy < {T} ? in0_p[bh * {T} + iy] : 0.f;
        float a = x, b = y;
#pragma unroll
        for (int d = 1; d < 32; d *= 2) {{
            float aa = __shfl_up_sync(0xffffffff, a, d), bb = __shfl_up_sync(0xffffffff, b, d);
            if (lane >= d) {{
                a = __fadd_rn(a, aa);
                b = __fadd_rn(b, bb);
            }}
        }}
        if (lane == 31) {{
            sums[warp] = a;
            sums[warp + 4] = b;
        }}
        __syncthreads();
        float carry = 0.f;
        if (warp >= 1)
            carry = sums[0];
        if (warp >= 2)
            carry = __fadd_rn(carry, sums[1]);
        if (warp >= 3)
            carry = __fadd_rn(carry, sums[2]);
        if (warp != 0)
            a = __fadd_rn(a, carry);
        float full = __fadd_rn(__fadd_rn(__fadd_rn(sums[0], sums[1]), sums[2]), sums[3]);
        if (warp >= 1)
            full = __fadd_rn(full, sums[4]);
        if (warp >= 2)
            full = __fadd_rn(full, sums[5]);
        if (warp >= 3)
            full = __fadd_rn(full, sums[6]);
        b = __fadd_rn(b, full);
        if (ix < {T})
            out0_p[bh * {T} + ix] = __fadd_rn(total, a);
        if (iy < {T})
            out0_p[bh * {T} + iy] = __fadd_rn(total, b);
        __syncthreads();
        float v = __fadd_rn(x, y);
#pragma unroll
        for (int d = 16; d > 0; d /= 2)
            v = __fadd_rn(v, __shfl_xor_sync(0xffffffff, v, d));
        if (lane == 0)
            sums[warp] = v;
        __syncthreads();
        float u = lane < 4 ? sums[lane] : 0.f;
        u = __fadd_rn(u, __shfl_xor_sync(0xffffffff, u, 2));
        u = __fadd_rn(u, __shfl_xor_sync(0xffffffff, u, 1));
        total = __fadd_rn(total, __shfl_sync(0xffffffff, u, 0));
        __syncthreads();
    }}
}}
clock_scan<<<{B * H}, 128>>>(@ARGS);
"""
        return jt.code(x.shape, "float32", [x], cuda_src=src)

    def grad(self, dy):
        return jt.flip(jt.flip(dy, [-1]).cumsum(-1), [-1])


def clock_cumsum(x):
    return ClockScan.apply(x)


class ClockTotal(jt.Function):
    def execute(self, x):
        self.shape = tuple(x.shape)
        B, H, T = x.shape
        src = f"""

__global__ void clock_total(@ARGS_DEF) {{
    @PRECALC
    int bh = blockIdx.x, tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    float a = 0.f, b = 0.f, c = 0.f, d = 0.f;
    for (int t = tid * 4; t < {T}; t += 1024) {{
        float4 v = *(const float4 *)(in0_p + bh * {T} + t);
        a += v.x;
        b += v.y;
        c += v.z;
        d += v.w;
    }}
    float sum = (a + b) + (c + d);
    for (int s = 16; s > 0; s /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, s);
    __shared__ float tmp[8];
    if (lane == 0)
        tmp[warp] = sum;
    __syncthreads();
    if (warp == 0) {{
        sum = lane < 8 ? tmp[lane] : 0.f;
        for (int s = 16; s > 0; s /= 2)
            sum += __shfl_down_sync(0xffffffff, sum, s);
        if (lane == 0)
            out0_p[bh] = sum;
    }}
}}
clock_total<<<{B * H}, 256>>>(@ARGS);
"""
        if T % 4 != 0:
            return x.sum(-1)
        return jt.code([B, H], "float32", [x], cuda_src=src)

    def grad(self, dy):
        return dy.unsqueeze(-1).broadcast(self.shape)


def clock_total(x):
    return ClockTotal.apply(x)


def _move(x, reverse):
    if reverse:
        B, H, T, D = x.shape
    else:
        B, T, H, D = x.shape
    if D % 8:
        raise ValueError("head layout requires D divisible by 8")
    shape = [B, T, H, D] if reverse else [B, H, T, D]
    src = f"""

__global__ void layout_copy(@ARGS_DEF) {{
    @PRECALC
    long long p = (long long)blockIdx.x * 256 + threadIdx.x;
    if (p < {B * H * T * D // 8}) {{
        int c = p % {D // 8}, row = p / {D // 8}, t = row % {T},
            bh = row / {T}, h = bh % {H}, b = bh / {H};
        long long head = ((long long)bh * {T} + t) * {D // 8} + c,
                  seq = (((long long)b * {T} + t) * {H} + h) * {D // 8} + c;
        ((int4 *)out0_p)[{"seq" if reverse else "head"}] = ((const int4 *)in0_p)[{"head" if reverse else "seq"}];
    }}
}}
layout_copy<<<({B * H * T * D // 8} + 255) / 256, 256>>>(@ARGS);
"""
    return jt.code(shape, "bfloat16", [x], cuda_src=src)


class HeadLayout(jt.Function):
    def execute(self, x, reverse):
        self.reverse = reverse
        return _move(x, reverse)

    def grad(self, dy):
        return _move(dy.bfloat16(), not self.reverse), None


def head_layout(x, reverse=False):
    return HeadLayout.apply(x, reverse)


class HeadScalarLayout(jt.Function):
    """FP32 BTH/BHT layout boundary with a coalesced head-major index."""

    def execute(self, x, reverse=False):
        self.reverse = reverse
        if reverse:
            B, H, T = x.shape
        else:
            B, T, H = x.shape
        self.dims = B, T, H
        return self.move(x, reverse)

    @staticmethod
    def move(x, reverse):
        if reverse:
            B, H, T = x.shape
        else:
            B, T, H = x.shape
        src = f"""

__global__ void scalar_layout(@ARGS_DEF) {{
    @PRECALC
    int p = blockIdx.x * 256 + threadIdx.x;
    if (p < {B * H * T}) {{
        int t = p % {T}, bh = p / {T};
        long long seq =
            ((long long)(bh / {H}) * {T} + t) * {H} + bh % {H};
        out0_p[{"seq" if reverse else "p"}] = in0_p[{"p" if reverse else "seq"}];
    }}
}}
scalar_layout<<<({B * H * T} + 255) / 256, 256>>>(@ARGS);
"""
        return jt.code(
            [B, T, H] if reverse else [B, H, T], "float32", [x], cuda_src=src
        )

    def grad(self, dy):
        return self.move(dy.float32(), not self.reverse), None


def head_scalar_layout(x):
    return HeadScalarLayout.apply(x, False)


class HeadGates(jt.Function):
    """One native FP32 permutation for the three public gate inputs."""

    def execute(self, g, delta, beta):
        self.dims = tuple(g.shape)
        B, T, H = self.dims
        assert tuple(delta.shape) == tuple(beta.shape) == self.dims
        src = f"""

__global__ void gate_layout_three(@ARGS_DEF) {{
    @PRECALC
    int p = blockIdx.x * 256 + threadIdx.x;
    if (p < {B * H * T}) {{
        int t = p % {T}, bh = p / {T};
        long long seq =
            ((long long)(bh / {H}) * {T} + t) * {H} + bh % {H};
        out0_p[p] = in0_p[seq];
        out1_p[p] = in1_p[seq];
        out2_p[p] = in2_p[seq];
    }}
}}
gate_layout_three<<<({B * H * T} + 255) / 256, 256>>>(@ARGS);
"""
        return jt.code([[B, H, T]] * 3, ["float32"] * 3, [g, delta, beta], cuda_src=src)

    def grad(self, dg, ddelta, dbeta):
        B, T, H = self.dims
        incoming = (dg, ddelta, dbeta)
        inputs = []
        expressions = []
        for d in incoming:
            if d is None:
                expressions.append("0.f")
            else:
                expressions.append(f"in{len(inputs)}_p[p]")
                inputs.append(d.float32())
        if not inputs:
            return None, None, None
        src = f"""

__global__ void gate_layout_three_grad(@ARGS_DEF) {{
    @PRECALC
    int p = blockIdx.x * 256 + threadIdx.x;
    if (p < {B * H * T}) {{
        int t = p % {T}, bh = p / {T};
        long long seq =
            ((long long)(bh / {H}) * {T} + t) * {H} + bh % {H};
        out0_p[seq] = {expressions[0]};
        out1_p[seq] = {expressions[1]};
        out2_p[seq] = {expressions[2]};
    }}
}}
gate_layout_three_grad<<<({B * H * T} + 255) / 256, 256>>>(@ARGS);
"""
        return jt.code([[B, T, H]] * 3, ["float32"] * 3, inputs, cuda_src=src)


def head_gates(g, delta, beta):
    return HeadGates.apply(g, delta, beta)


def local_log2_cumsum(g):
    B, H, T = g.shape
    N = T // 64
    src = f"""

__global__ void local_scan(@ARGS_DEF) {{
    @PRECALC
    int row = blockIdx.x, tid = threadIdx.x, lane = tid % 32;
    __shared__ float carry;
    float v = in0_p[(long long)row * 64 + tid];
#pragma unroll
    for (int d = 1; d < 32; d *= 2) {{
        float u = __shfl_up_sync(0xffffffff, v, d);
        if (lane >= d)
            v = __fadd_rn(v, u);
    }}
    if (tid == 31)
        carry = v;
    __syncthreads();
    if (tid >= 32)
        v = __fadd_rn(v, carry);
    out0_p[(long long)row * 64 + tid] = v * 1.4426950408889634f;
}}
local_scan<<<{B * H * N}, 64>>>(@ARGS);
"""
    return jt.code(g.shape, "float32", [g], cuda_src=src)


def exp2(x):
    n = 1
    for v in x.shape:
        n *= v
    return jt.code(
        x.shape,
        "float32",
        [x],
        cuda_src=f"""

__global__ void exp2_kernel(@ARGS_DEF) {{ @PRECALC
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < {n})
        out0_p[i] = exp2f(in0_p[i]);
}}
exp2_kernel<<<({n} + 255) / 256, 256>>>(@ARGS);
""",
    )


class PhaseWriteOnly(jt.Function):
    def execute(self, lam, beta, M):
        rpw = 2
        rows = 8 * rpw
        if M != 128:
            raise ValueError((M, rows))
        self.lam, self.beta, self.M = lam, beta, M
        n = beta.numel()
        threads = 256
        src = f"""

__global__ void phase_write_more_rows(@ARGS_DEF) {{
    @PRECALC
    int lane = threadIdx.x % 32;
#pragma unroll
    for (int rowpart = 0; rowpart < {rpw}; rowpart++) {{
        int row = blockIdx.x * {rows} + (threadIdx.x / 32) * {rpw} + rowpart;
        if (row >= {n})
            continue;
        float lam = in0_p[row], whole = floorf(lam), fraction = lam - whole, beta = in1_p[row];
        int wi = ((int)whole) % 127;
        if (wi < 0)
            wi += 127;
#pragma unroll
        for (int part = 0; part < 2; part++) {{
            int pair = lane + part * 32;
            float coord = (float)((wi * pair) % 127) + fraction * pair;
            if (coord > 63.5f)
                coord -= 127.f;
            float phase = coord * 0.04947390005653217f, co = __cosf(phase), si = __sinf(phase);
            float scale = sqrtf((pair == 0 ? 1.f : 2.f) / 127.f);
            __nv_bfloat162 value = __floats2bfloat162_rn(beta * scale * co, -beta * scale * si);
            *(unsigned *)(out0_p + (long long)row * 128 + pair * 2) = *(unsigned *)&value;
        }}
    }}
}}
phase_write_more_rows<<<({n} + {rows - 1}) / {rows}, {threads}>>>(@ARGS);
"""
        return jt.code(
            list(lam.shape) + [128],
            "bfloat16",
            [lam, beta],
            cuda_header="#include <cuda_bf16.h>",
            cuda_src=src,
        )

    def grad(self, dw):
        beta = self.beta
        lam = self.lam
        M = self.M
        n = beta.numel()
        if dw is None:
            dw = jt.zeros(list(beta.shape) + [M])
        src = f"""

__global__ void phase_write_only_grad(@ARGS_DEF) {{
    @PRECALC
    int tid = threadIdx.x, row = blockIdx.x * 4 + tid / 64, pair = tid % 64, lane = tid % 32;
    float dl = 0.f, db = 0.f;
    if (row < {n}) {{
        long long p = (long long)row * 128 + pair * 2;
        float lam = in0_p[row], whole = floorf(lam), fraction = lam - whole;
        int wi = ((int)whole) % 127;
        if (wi < 0)
            wi += 127;
        float coord = (float)((wi * pair) % 127) + fraction * pair;
        if (coord > 63.5f)
            coord -= 127.f;
        float phase = coord * 0.04947390005653217f, c = __cosf(phase), s = __sinf(phase),
              b = in1_p[row], de = (float)in2_p[p], doo = (float)in2_p[p + 1];
        float scale = sqrtf((pair == 0 ? 1.f : 2.f) / 127.f), omega = pair * 0.04947390005653217f;
        db = de * (scale * c) + doo * (-scale * s);
        dl = (de * b * (-scale * s) + doo * b * (-scale * c)) * omega;
    }}
    for (int d = 16; d > 0; d /= 2) {{
        dl += __shfl_down_sync(0xffffffff, dl, d);
        db += __shfl_down_sync(0xffffffff, db, d);
    }}
    __shared__ float sl[8], sb[8];
    if (lane == 0) {{
        sl[tid / 32] = dl;
        sb[tid / 32] = db;
    }}
    __syncthreads();
    if (tid % 64 == 0 && row < {n}) {{
        out0_p[row] = sl[tid / 32] + sl[tid / 32 + 1];
        out1_p[row] = sb[tid / 32] + sb[tid / 32 + 1];
    }}
}}
phase_write_only_grad<<<({n} + 3) / 4, 256>>>(@ARGS);
"""
        dl, db = jt.code(
            [beta.shape, beta.shape],
            ["float32", "float32"],
            [lam, beta, dw],
            cuda_src=src,
        )
        return dl, db, None


def phase_write_compact(lam, beta, M):
    if M != 128:
        raise ValueError("SM80 phase kernel requires 128 slots")
    return PhaseWriteOnly.apply(lam, beta, M)


CLOCK_PHASE_HEADER = """

__device__ __forceinline__ float cyfa_clock_phase(float lam, int pair) {
    float whole = floorf(lam), fraction = lam - whole;
    int wi = ((int)whole) % 127;
    if (wi < 0)
        wi += 127;
    float coord = (float)((wi * pair) % 127) + fraction * pair;
    if (coord > 63.5f)
        coord -= 127.f;
    return coord * 0.04947390005653217f;
}
"""


class PhaseWrite(jt.Function):
    def execute(self, lam, beta, M):
        self.beta = beta
        self.M = M
        shape = list(lam.shape) + [M // 2]
        n = 1
        for x in lam.shape:
            n *= x
        src = f"""

__global__ void phase_write(@ARGS_DEF) {{
    @PRECALC
    int row = blockIdx.x * 4 + threadIdx.x / 64, pair = threadIdx.x % 64;
    if (row >= {n})
        return;
    float lam = in0_p[row], whole = floorf(lam), fraction = lam - whole;
    int wi = ((int)whole) % {M - 1};
    if (wi < 0)
        wi += {M - 1};
    float coord = (float)((wi * pair) % {M - 1}) + fraction * pair;
    if (coord > {(M - 1) * 0.5})
        coord -= {M - 1};
    float phase = coord * {6.283185307179586 / (M - 1)}f, co = __cosf(phase), si = __sinf(phase);
    float scale = sqrtf((pair == 0 ? 1.f : 2.f) / {M - 1});
    out1_p[(long long)row * 64 + pair] = co;
    out2_p[(long long)row * 64 + pair] = si;
    __nv_bfloat162 value = __floats2bfloat162_rn(in1_p[row] * scale * co, -in1_p[row] * scale * si);
    *(unsigned *)(out0_p + (long long)row * 128 + pair * 2) = *(unsigned *)&value;
}}
phase_write<<<({n} + 3) / 4, 256>>>(@ARGS);
"""
        write, self.c, self.s = jt.code(
            [list(lam.shape) + [M], shape, shape],
            ["bfloat16", "float32", "float32"],
            [lam, beta],
            cuda_header="#include <cuda_bf16.h>",
            cuda_src=src,
        )
        return write, self.c, self.s

    def grad(self, dw, dc, ds):
        beta = self.beta
        M = self.M
        n = 1
        for x in beta.shape:
            n *= x
        if dc is None:
            dc = jt.zeros_like(self.c)
        if ds is None:
            ds = jt.zeros_like(self.s)
        if dw is None:
            dw = jt.zeros(list(beta.shape) + [M])
        src = f"""

__global__ void phase_grad(@ARGS_DEF) {{
    @PRECALC
    int tid = threadIdx.x, row = blockIdx.x * 4 + tid / 64, pair = tid % 64, lane = tid % 32;
    float dl = 0.f, db = 0.f;
    if (row < {n}) {{
        long long ph = (long long)row * 64 + pair, p = (long long)row * 128 + pair * 2;
        float c = in0_p[ph], s = in1_p[ph], b = in2_p[row], de = (float)in3_p[p],
              doo = (float)in3_p[p + 1];
        float scale = sqrtf((pair == 0 ? 1.f : 2.f) / {M - 1}), omega = pair * {6.283185307179586 / (M - 1)}f;
        db = de * (scale * c) + doo * (-scale * s);
        dl = (de * b * (-scale * s) + doo * b * (-scale * c) + in4_p[ph] * (-s) + in5_p[ph] * c) *
             omega;
    }}
    for (int d = 16; d > 0; d /= 2) {{
        dl += __shfl_down_sync(0xffffffff, dl, d);
        db += __shfl_down_sync(0xffffffff, db, d);
    }}
    __shared__ float sl[8], sb[8];
    if (lane == 0) {{
        sl[tid / 32] = dl;
        sb[tid / 32] = db;
    }}
    __syncthreads();
    if (tid % 64 == 0 && row < {n}) {{
        out0_p[row] = sl[tid / 32] + sl[tid / 32 + 1];
        out1_p[row] = sb[tid / 32] + sb[tid / 32 + 1];
    }}
}}
phase_grad<<<({n} + 3) / 4, 256>>>(@ARGS);
"""
        dl, db = jt.code(
            [beta.shape, beta.shape],
            ["float32", "float32"],
            [self.c, self.s, beta, dw, dc, ds],
            cuda_src=src,
        )
        return dl, db, None


def phase_write(lam, beta, M, compact=None):
    # Internal callers that need c/s explicitly opt out. The prepared-core API
    # has no q/v dimensions, so its default selection uses the large threshold.
    if compact is None:
        compact = M == 128 and use_large(lam)
    if compact:
        return phase_write_compact(lam, beta, M), lam, None
    if M != 128:
        raise ValueError("SM80 phase kernel requires 128 slots")
    return PhaseWrite.apply(lam, beta, M)


def _rot(x, c, s, negative=False):
    p = x.reshape(list(x.shape[:-1]) + [x.shape[-1] // 2, 2])
    e, o = p[..., 0], p[..., 1]
    if negative:
        a, b = c * e + s * o, -s * e + c * o
    else:
        a, b = c * e - s * o, s * e + c * o
    return jt.stack([a, b], -1).reshape(x.shape)


class ModalReadout(jt.Function):
    def execute(self, raw, c, s, table):
        B, H, T, M = raw.shape
        R = M - 1
        if M != 128:
            raise ValueError("WMMA readout currently requires M=128 and BF16")
        self.raw, self.c, self.s, self.table = raw, c, s, table
        self.dims = B, H, T, M
        src = f"""

__global__ void readout(@ARGS_DEF) {{
    @PRECALC
    using namespace nvcuda;
    int bh = blockIdx.y, h = bh % {H}, t0 = blockIdx.x * 48, tid = threadIdx.x,
        warp = tid / 32, lane = tid % 32;
    extern __shared__ __align__(32) unsigned char smem[];
    __nv_bfloat16 *w = (__nv_bfloat16 *)smem, *a = w + 128 * 136;
    float *work = (float *)(a + 48 * 136);
    for (int p = tid; p < 128 * 128; p += 256) {{
        int r = p / 128, m = p % 128;
        w[r * 136 + m] =
            __float2bfloat16(r < 127 ? in3_p[(long long)h * 127 * 128 + r * 128 + m] : 0.f);
    }}
    for (int p = tid; p < 48 * 128; p += 256) {{
        int row = p / 128, m = p % 128, t = t0 + row;
        float z = 0.f;
        if (t < {T}) {{
            long long q = ((long long)bh * {T} + t) * 128 + m,
                      ph = ((long long)bh * {T} + t) * 64 + m / 2;
            float e = (float)in0_p[q - (m % 2)], o = (float)in0_p[q - (m % 2) + 1];
            z = m % 2 ? in2_p[ph] * e + in1_p[ph] * o : in1_p[ph] * e - in2_p[ph] * o;
        }}
        a[row * 136 + m] = __float2bfloat16(z);
    }}
    __syncthreads();
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[3];
    for (int j = 0; j < 3; j++)
        wmma::fill_fragment(acc[j], 0.f);
#pragma unroll
    for (int k = 0; k < 128; k += 16) {{
        cyfa_fast_load(bf, w + warp * 16 * 136 + k, 136);
        for (int j = 0; j < 3; j++) {{
            cyfa_fast_load(af, a + j * 16 * 136 + k, 136);
            wmma::mma_sync(acc[j], af, bf, acc[j]);
        }}
    }}
    for (int j = 0; j < 3; j++)
        wmma::store_matrix_sync(work + j * 16 * 136 + warp * 16, acc[j], 136, wmma::mem_row_major);
    __syncthreads();
    for (int row = warp; row < 48; row += 8) {{
        float mx = -1.e30f;
        for (int col = lane; col < 128; col += 32)
            if (col < 127)
                mx = fmaxf(mx, work[row * 136 + col]);
        for (int d = 16; d > 0; d /= 2)
            mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, d));
        mx = __shfl_sync(0xffffffff, mx, 0);
        float sum = 0.f;
        for (int col = lane; col < 128; col += 32) {{
            float p = col < 127 ? expf(work[row * 136 + col] - mx) : 0.f;
            work[row * 136 + col] = p;
            sum += p;
        }}
        for (int d = 16; d > 0; d /= 2)
            sum += __shfl_down_sync(0xffffffff, sum, d);
        sum = __shfl_sync(0xffffffff, sum, 0);
        for (int col = lane; col < 128; col += 32) {{
            float p = work[row * 136 + col] / sum;
            a[row * 136 + col] = __float2bfloat16(p);
            if (t0 + row < {T} && col < 127)
                out1_p[((long long)bh * {T} + t0 + row) * 127 + col] = p;
        }}
    }}
    __syncthreads();
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bf2;
    for (int j = 0; j < 3; j++)
        wmma::fill_fragment(acc[j], 0.f);
#pragma unroll
    for (int k = 0; k < 128; k += 16) {{
        cyfa_fast_load(bf2, w + k * 136 + warp * 16, 136);
        for (int j = 0; j < 3; j++) {{
            cyfa_fast_load(af, a + j * 16 * 136 + k, 136);
            wmma::mma_sync(acc[j], af, bf2, acc[j]);
        }}
    }}
    for (int j = 0; j < 3; j++)
        wmma::store_matrix_sync(work + j * 16 * 136 + warp * 16, acc[j], 136, wmma::mem_row_major);
    __syncthreads();
    for (int p = tid; p < 48 * 128; p += 256) {{
        int row = p / 128, m = p % 128, t = t0 + row;
        if (t < {T}) {{
            long long q = ((long long)bh * {T} + t) * 128 + m,
                      ph = ((long long)bh * {T} + t) * 64 + m / 2;
            float e = work[row * 136 + m - (m % 2)], o = work[row * 136 + m - (m % 2) + 1];
            out0_p[q] =
                (out0_type)(m % 2 ? -in2_p[ph] * e + in1_p[ph] * o : in1_p[ph] * e + in2_p[ph] * o);
            out2_p[q] = work[row * 136 + m];
        }}
    }}
}}
cudaFuncSetAttribute(readout, cudaFuncAttributeMaxDynamicSharedMemorySize, 73984);
readout<<<dim3(({T} + 47) / 48, {B * H}), 256, 73984>>>(@ARGS);
"""
        weights, self.probs, self.coeff = jt.code(
            [raw.shape, [B, H, T, R], raw.shape],
            ["bfloat16", "float32", "float32"],
            [raw, c, s, table],
            cuda_header=WMMA_HEADER,
            cuda_src=src,
        )
        return weights

    def grad(self, dw):
        B, H, T, M = self.dims
        raw, c, s, table = self.raw.bfloat16(), self.c, self.s, self.table
        dw = dw.bfloat16()

        def head(x):
            return x.transpose(1, 0, 2, 3).reshape((H, B * T, x.shape[-1]))

        def unhead(x):
            return x.reshape((H, B, T, x.shape[-1])).transpose(1, 0, 2, 3)

        du = _rot(dw.float32(), c, s)
        zfull = head(_rot(raw.float32(), c, s))
        duh32 = head(du)
        scores = _mm(zfull.bfloat16(), table.bfloat16(), tb=True)
        ph = jt.nn.softmax(scores, dim=-1)
        duh = duh32.bfloat16()
        tb = table.bfloat16()
        dp = _mm(duh32.bfloat16(), table.bfloat16(), tb=True)
        dl = ((dp - (dp * ph).sum(-1, keepdims=True)) * ph).bfloat16()
        dz = unhead(bmm(dl, tb))
        z = head(_rot(raw.float32(), c, s)).bfloat16()
        dt = bmm(ph.bfloat16(), duh, ta=True) + bmm(dl, z, ta=True)
        p = dw.float32().reshape((B, H, T, M // 2, 2))
        u = unhead(bmm(ph.bfloat16(), tb)).reshape((B, H, T, M // 2, 2))
        dcz = p[..., 0] * u[..., 0] + p[..., 1] * u[..., 1]
        dsz = p[..., 0] * u[..., 1] - p[..., 1] * u[..., 0]
        zz = dz.reshape((B, H, T, M // 2, 2))
        r = raw.float32().reshape((B, H, T, M // 2, 2))
        dc = dcz + zz[..., 0] * r[..., 0] + zz[..., 1] * r[..., 1]
        ds = dsz - zz[..., 0] * r[..., 1] + zz[..., 1] * r[..., 0]
        dr = _rot(dz, c, s, True).bfloat16()
        return dr, dc, ds, dt


def modal_readout(raw, c, s, table):
    return ModalReadout.apply(raw, c, s, table)
