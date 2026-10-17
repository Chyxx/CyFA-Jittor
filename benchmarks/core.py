"""Prepared-input complete CyFA core benchmark, explicit BF16 VJP, complete trace integrity and independent core scope.

The prepared-input adapters preserve each backend's complete core computation.

Run --check-source for CPU-only adapter review. Actual prepare/torch/jt commands
require an explicitly selected GPU and are never run by the CPU source check.
Outputs and caches are confined to the configured benchmark directory.
"""

from __future__ import annotations

import argparse
import ast
import collections
import csv
import ctypes
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEYS = ("q", "k", "v", "g", "delta", "beta", "table")
CONTRACT = "prepared_core_cyfa_v1"
ADAPTER_NORM = r"""
# Adapter-only preparation boundary: these tensors were prepared before timing.
def qk_rmsnorm_fwd(q,k,q_weight,k_weight,q_eps,k_eps,*,store_q=True,store_rstd=False):
    return q,k,_CORE_RSTD_ONES,_CORE_RSTD_ONES

def _qk_rmsnorm_bwd(dq_out,dk_out,q,k,q_weight,k_weight,q_rstd,k_rstd):
    return dq_out,dk_out,None,None
"""


def resolve_path(path):
    p = Path(path).expanduser().resolve()
    return p


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False))


def build_adapter(baseline_root, output_dir):
    """AST extraction and Python compilation only; no torch/CUDA import here."""
    source_path = resolve_path(baseline_root) / "fla/ops/cyfa/chunk.py"
    source = source_path.read_text()
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    names = (
        "_scalar_decay_fwd",
        "_scalar_decay_fused_readout_fwd",
        "ChunkCyFAFunction",
    )
    extracted = {}
    changed = {}
    locations = {}
    for name in names:
        node = next(
            x
            for x in tree.body
            if isinstance(x, (ast.ClassDef, ast.FunctionDef)) and x.name == name
        )
        start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
        text = "".join(lines[start : node.end_lineno])
        extracted[name] = text
        locations[name] = {
            "line": start + 1,
            "original_sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
        if name != "ChunkCyFAFunction":
            old = "final_state = final_state.transpose(-1, -2).contiguous()"
            if text.count(old) != 1:
                raise AssertionError((name, "expected exactly one final-state copy"))
            text = text.replace(old, "final_state = final_state.transpose(-1, -2)")
        changed[name] = text
    # The class, its decorators, backward math and all kernel objects stay intact.
    if changed["ChunkCyFAFunction"] != extracted["ChunkCyFAFunction"]:
        raise AssertionError("class changed")
    adapted = ADAPTER_NORM + "\n\n" + "\n\n".join(changed[n] for n in names) + "\n"
    compile(adapted, "prepared_core_adapter", "exec")
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = hashlib.sha256(str(source_path).encode()).hexdigest()[:12]
    original_out = output_dir / f"original-source-{tag}.py"
    adapter_out = output_dir / f"prepared-adapter-{tag}.py"
    original_out.write_text(source)
    adapter_out.write_text(adapted)
    audit = {
        "contract": CONTRACT,
        "baseline_source_path": str(source_path),
        "baseline_source_sha256": sha(source_path),
        "saved_original_path": str(original_out),
        "saved_original_sha256": sha(original_out),
        "adapter_path": str(adapter_out),
        "adapter_sha256": sha(adapter_out),
        "symbols": locations,
        "class_copied_verbatim": True,
        "helper_rewrites": {
            "_scalar_decay_fwd": 1,
            "_scalar_decay_fused_readout_fwd": 1,
        },
        "norm_policy": "Return prepared q/k and the preallocated FP32 rstd=ones; backward returns dq/dk and no norm-weight grads.",
        "final_state_policy": "Only final_state.transpose(...).contiguous() becomes transpose view; no core math/recompute changes.",
        "unchanged": "Original class decorators, casts, zeroing, phase/scan/readout/table backward, DS/DG, H recomputation and Triton kernel objects.",
        "scope": "Zero initial state only: Torch h0=None, native initial zeros are preallocated outside timing.",
    }
    write_json(output_dir / f"adapter-audit-{tag}.json", audit)
    return adapted, audit


def import_baseline(baseline_root):
    baseline_root = resolve_path(baseline_root)
    os.environ["CYFA_BASELINE_ROOT"] = str(baseline_root)
    sys.path.insert(0, str(baseline_root))
    sys.path.insert(0, str(ROOT / "benchmarks"))
    import torch
    import triton

    if triton.__version__ != "3.2.0":
        raise AssertionError(
            "baseline must use fla Triton 3.2.0, got " + triton.__version__
        )
    from support import load_baseline

    load_baseline()
    import importlib

    module = importlib.import_module("fla.ops.cyfa.chunk")
    if Path(module.__file__).resolve() != baseline_root / "fla/ops/cyfa/chunk.py":
        raise AssertionError(("unexpected baseline module", module.__file__))
    return torch, triton, module


def prepare(args):
    """One-time common preparation; all dtype/layout/norm/table work is excluded."""
    import numpy as np

    torch, triton, module = import_baseline(args.baseline_root)
    from fla.ops.cyfa.qk_norm import qk_rmsnorm_fwd
    from fla.ops.cyfa.utils import build_readout_table

    fixture = resolve_path(
        args.original_fixture or ROOT / "results/benchmark" / f"fixture-{args.case}.npz"
    )
    with np.load(fixture) as z:

        def tensor(key, dtype):
            return torch.tensor(
                np.ascontiguousarray(z[key]), device="cuda", dtype=dtype
            )

        q, k = tensor("q", torch.bfloat16), tensor("k", torch.bfloat16)
        qw, kw = (
            tensor("q_norm_weight", torch.float32),
            tensor("k_norm_weight", torch.float32),
        )
        with torch.no_grad():
            qn, kn, _, _ = qk_rmsnorm_fwd(
                q, k, qw, kw, 1e-6, 1e-6, store_q=True, store_rstd=True
            )
            table = build_readout_table(tensor("readout", torch.float32))
        torch.cuda.synchronize()
        values = {
            key: np.ascontiguousarray(z[key], dtype=np.float32)
            for key in ("g", "delta", "beta")
        }
        # The old fixture may store raw FP32 v/do; freeze their BF16 values once.
        for key in ("v", "do"):
            values[key] = tensor(key, torch.bfloat16).float().cpu().numpy()
        values.update(
            q=qn.float().cpu().numpy(),
            k=kn.float().cpu().numpy(),
            table=table.float().cpu().numpy(),
        )
    B, T, H, D = values["q"].shape
    M = values["table"].shape[-1]
    if (
        (B, T) not in ((4, 2048), (32, 2048), (4, 8192))
        or H != 4
        or D != 256
        or M != 128
    ):
        raise AssertionError((B, T, H, D, M))
    out = resolve_path(args.prepared or args.output_dir / f"prepared-{args.case}.npz")
    np.savez(out, **values)
    meta = {
        "contract": CONTRACT,
        "case": args.case,
        "prepared_path": str(out),
        "prepared_sha256": sha(out),
        "original_fixture_path": str(fixture),
        "original_fixture_sha256": sha(fixture),
        "shape": [B, T, H, D, M],
        "layout": "q/k/v/do B,T,H,D; gates B,T,H; table H,127,128",
        "storage": "FP32 NPZ; q/k/v/do values are exact BF16 representable values.",
        "norm_preparation": "Unmodified FLA qk_rmsnorm_fwd store_q=True outside timing.",
        "table_preparation": "Unmodified build_readout_table(readout) outside timing; table is the core leaf.",
        "torch": torch.__version__,
        "triton": triton.__version__,
        "qk_norm_source_sha256": sha(
            Path(args.baseline_root) / "fla/ops/cyfa/qk_norm.py"
        ),
        "utils_source_sha256": sha(Path(args.baseline_root) / "fla/ops/cyfa/utils.py"),
    }
    write_json(out.with_suffix(".meta.json"), meta)
    print(json.dumps(meta), flush=True)


class TorchRuntime:
    def __init__(self, args, arrays, adapter, audit):
        import numpy as np

        self.torch, self.triton, module = import_baseline(args.baseline_root)
        torch = self.torch
        self.args = args
        self.audit = audit
        self.x = {
            key: torch.tensor(
                np.ascontiguousarray(arrays[key]),
                device="cuda",
                dtype=torch.bfloat16 if key in ("q", "k", "v") else torch.float32,
                requires_grad=True,
            )
            for key in KEYS
        }
        self.do = torch.tensor(
            np.ascontiguousarray(arrays["do"]), device="cuda", dtype=torch.bfloat16
        )
        B, T, H, D = self.x["q"].shape
        self.norm_weight = torch.ones(D, device="cuda", dtype=torch.float32)
        self.rstd = torch.ones(B * T * H, device="cuda", dtype=torch.float32)
        # Private exec namespace: the imported module and repository stay unchanged.
        ns = dict(module.__dict__)
        ns["_CORE_RSTD_ONES"] = self.rstd
        exec(compile(adapter, audit["adapter_path"], "exec"), ns, ns)
        self.function = ns["ChunkCyFAFunction"]
        self.adapter_namespace = ns
        self.sync()

    def sync(self):
        self.torch.cuda.synchronize()

    def forward(self, inference=False):
        x = self.x

        def call():
            return self.function.apply(
                x["q"],
                x["k"],
                x["v"],
                x["g"],
                x["delta"],
                x["beta"],
                x["table"],
                self.norm_weight,
                self.norm_weight,
                1e-6,
                1e-6,
                x["q"].shape[-1] ** -0.5,
                None,
                None,
                None,
                True,
                None,
                None,
                1 if inference else 0,
            )

        if inference:
            with self.torch.no_grad():
                out = call()
        else:
            out = call()
        # The custom Function returns four tensors; expose the common core API.
        return out[0], (out[1], out[2], out[3])

    def prepare_step(self, mode):
        if mode == "backward":
            self.prepared_forward = self.forward()
            self.sync()

    def run(self, mode):
        y, states = (
            self.prepared_forward
            if mode == "backward"
            else self.forward(inference=mode == "inference")
        )
        if mode in ("forward_backward", "backward"):
            grads = self.torch.autograd.grad(y, list(self.x.values()), self.do)
            return y, states, grads
        return y, states

    def array(self, value):
        return value.detach().float().cpu().numpy()

    def canonical(self, value, key):
        return self.array(value)

    def collect(self):
        pass

    def provenance(self):

        root = Path(self.args.baseline_root)
        paths = [
            root / "fla/ops/cyfa/chunk.py",
            root / "fla/ops/cyfa/qk_norm.py",
            root / "fla/ops/cyfa/utils.py",
            root / "fla/ops/common/chunk_h.py",
            root / "fla/ops/common/chunk_o.py",
            ROOT / "benchmarks/support.py",
        ]
        return {
            "torch": self.torch.__version__,
            "triton": self.triton.__version__,
            "source_sha256": {str(p): sha(p) for p in paths},
            "adapter": self.audit,
            "identity_rstd_preallocated": True,
            "norm_weights_require_grad": False,
            "core_leaf_names": list(KEYS),
            "initial_state_policy": "Torch None, final transpose view only",
        }


class JittorRuntime:
    def __init__(self, args, arrays):
        import numpy as np

        sys.path.insert(0, str(resolve_path(args.source_root)))
        import jittor as jt

        jt.flags.use_cuda = 1
        jt.flags.use_tensorcore = 0
        self.jt = jt
        self.args = args
        import cyclic_flow_attention
        from cyclic_flow_attention.ops.chunk import scalar_la
        from cyclic_flow_attention.ops.chunk_readout import first_readout
        from cyclic_flow_attention.ops.utils import (
            clock_cumsum,
            clock_total,
            phase_write,
        )

        self.package_path = Path(cyclic_flow_attention.__file__).resolve().parent
        if (
            self.package_path
            != resolve_path(args.source_root) / "cyclic_flow_attention"
        ):
            raise AssertionError(self.package_path)
        self.clock_cumsum = clock_cumsum
        self.clock_total = clock_total
        self.phase_write = phase_write
        self.first_readout = first_readout
        self.scalar_la = scalar_la

        def layout(value, key):
            if key in ("q", "k", "v", "do"):
                return value.transpose(0, 2, 1, 3)
            if key in ("g", "delta", "beta"):
                return value.transpose(0, 2, 1)
            return value

        self.x = {}
        for key in KEYS:
            value = jt.array(np.ascontiguousarray(layout(arrays[key], key))).cast(
                "bfloat16" if key in ("q", "k", "v") else "float32"
            )
            value.start_grad()
            self.x[key] = value
        self.do = (
            jt.array(np.ascontiguousarray(layout(arrays["do"], "do")))
            .bfloat16()
            .stop_grad()
        )

        class UnitCotangent(jt.Function):
            def execute(self, y, do_bf16):
                self.do = do_bf16
                # This dummy is only an AD root. Its numerical value is unused.
                return jt.zeros((1,), dtype="float32")

            def grad(self, dloss):
                # Benchmark-only unit seed: jt.grad(dummy,leaves) always injects 1.
                # Deliberately return the already prepared BF16 cotangent directly;
                # do not use this adapter for arbitrary scaled/higher-order losses.
                return self.do, None

        self.unit_cotangent = UnitCotangent.apply
        B, H, T, K = self.x["q"].shape
        M = self.x["table"].shape[-1]
        V = self.x["v"].shape[-1]
        self.hk = jt.zeros((B, H, K, M), dtype="float32").stop_grad()
        self.hv = jt.zeros((B, H, M, V), dtype="float32").stop_grad()
        # No timed input cast, transpose, normalization, basis/table construction.
        self.sync()

    def sync(self):
        self.jt.sync_all(True)

    def forward(self, inference=False):
        def call():
            x = self.x
            lam = self.clock_cumsum(x["delta"])
            write, c, s = self.phase_write(lam, x["beta"], 128)
            weights, hkt = self.first_readout(
                x["q"],
                x["k"],
                write,
                x["g"],
                self.hk,
                c,
                s,
                x["table"],
                x["q"].shape[-1] ** -0.5,
            )
            y, hvt = self.scalar_la(weights, write, x["v"], x["g"], self.hv, 1.0, False)
            return y, (hkt, hvt, self.clock_total(x["delta"]))

        if inference:
            with self.jt.no_grad():
                return call()
        return call()

    def prepare_step(self, mode):
        if mode == "backward":
            self.prepared_forward = self.forward()
            self.sync()

    def run(self, mode):
        y, states = (
            self.prepared_forward
            if mode == "backward"
            else self.forward(inference=mode == "inference")
        )
        if mode in ("forward_backward", "backward"):
            grads = self.jt.grad(self.unit_cotangent(y, self.do), list(self.x.values()))
            # Required: keep every forward output alive through synchronization.
            return y, states, grads
        return y, states

    def array(self, value):
        return value.float32().numpy()

    def canonical(self, value, key):
        a = self.array(value)
        if key in ("out", "q", "k", "v"):
            return a.transpose(0, 2, 1, 3)
        if key in ("g", "delta", "beta"):
            return a.transpose(0, 2, 1)
        return a

    def collect(self):
        self.jt.gc()

    def provenance(self):
        return {
            "jittor": self.jt.__version__,
            "package_path": str(self.package_path),
            "source_sha256": {
                str(p): sha(p) for p in sorted(self.package_path.rglob("*.py"))
            },
            "core_leaf_names": list(KEYS),
            "initial_state_policy": "Preallocated zero FP32 HK/HV, no timed layout",
            "runtime_imports": {
                "torch": "torch" in sys.modules,
                "triton": "triton" in sys.modules,
            },
        }


def export_validation(runtime, args):
    import numpy as np

    result = runtime.run("forward_backward")
    runtime.sync()
    values = {"out": np.ascontiguousarray(runtime.canonical(result[0], "out"))}
    for i, s in enumerate(result[1]):
        values["state" + str(i)] = np.ascontiguousarray(runtime.array(s))
    for key, g in zip(KEYS, result[2]):
        values["grad_" + key] = np.ascontiguousarray(runtime.canonical(g, key))
    del result
    path = (
        args.output_dir / f"full-{args.case}-{args.framework}-{args.label}-values.npz"
    )
    np.savez(path, **values)
    repeat = []
    repeat_values = []
    repeat_metrics = []
    for _ in range(3):
        result = runtime.run("forward_backward")
        runtime.sync()
        trial = {"out": np.ascontiguousarray(runtime.canonical(result[0], "out"))}
        for i, state in enumerate(result[1]):
            trial["state" + str(i)] = np.ascontiguousarray(runtime.array(state))
        for key, grad in zip(KEYS, result[2]):
            trial["grad_" + key] = np.ascontiguousarray(runtime.canonical(grad, key))
        errors = {
            key: float(np.max(np.abs(trial[key] - values[key]))) for key in values
        }
        metrics = {}
        for key in values:
            ref = values[key].astype("float64")
            x = trial[key].astype("float64")
            metrics[key] = {
                "rrms": float(
                    np.sqrt(np.mean((x - ref) ** 2))
                    / max(np.sqrt(np.mean(ref**2)), 1e-12)
                ),
                "max_abs_vs_reference_max": errors[key]
                / max(float(np.max(np.abs(ref))), 1e-12),
            }
        repeat_metrics.append(metrics)
        repeat_values.append(errors)
        repeat.append(max(errors.values()))
        del result, trial
    finite = {k: bool(np.isfinite(v).all()) for k, v in values.items()}
    runtime.collect()
    numerical = {
        "export_path": str(path),
        "finite": finite,
        "repeat_max_abs": repeat,
        "repeat_values_max_abs": repeat_values,
        "repeat_scope": "out, three final states and all seven core leaf gradients",
        "repeat_metrics": repeat_metrics,
        "repeat_tolerance": {"rrms": 1e-6, "max_abs_vs_reference_max": 1e-5},
        "repeat_bitwise_equal": not any(repeat),
        "export_gate": "PASS"
        if all(finite.values())
        and all(
            m["rrms"] <= 1e-6 and m["max_abs_vs_reference_max"] <= 1e-5
            for trial in repeat_metrics
            for m in trial.values()
        )
        else "FAIL",
        "gate": "PENDING_COMPARE",
        "comparison_status": "Not compared to another backend yet.",
    }
    if args.reference:
        numerical["comparison"] = compare_values(path, resolve_path(args.reference))
        numerical["gate"] = numerical["comparison"]["gate"]
    write_json(
        args.output_dir
        / f"full-{args.case}-{args.framework}-{args.label}-validation.json",
        numerical,
    )
    if numerical["export_gate"] != "PASS" or numerical["gate"] == "FAIL":
        raise RuntimeError("Numerical/determinism gate failed: " + str(numerical))
    return numerical


def compare_values(actual, reference):
    import numpy as np

    errors = {}
    fail = []
    with np.load(actual) as a, np.load(reference) as b:
        required = ["out", "state0", "state1", "state2"] + ["grad_" + k for k in KEYS]
        for name in required:
            x = a[name].astype(np.float32)
            ref = b[name].astype(np.float32)
            if x.shape != ref.shape:
                raise AssertionError((name, x.shape, ref.shape))
            if not np.isfinite(ref).all():
                raise ValueError("Nonfinite reference: " + name)
            rrms = float(
                np.sqrt(np.mean((x - ref) ** 2)) / max(np.sqrt(np.mean(ref**2)), 1e-12)
            )
            tol = (
                0.02
                if name in ("grad_g", "grad_delta", "grad_beta")
                else (0.01 if name.startswith("grad_") else 0.005)
            )
            errors[name] = {
                "rrms": rrms,
                "max_abs": float(np.max(np.abs(x - ref))),
                "tolerance": tol,
                "finite": bool(np.isfinite(x).all()),
            }
            if not errors[name]["finite"] or rrms >= tol:
                fail.append(name)
    return {
        "actual": str(actual),
        "reference": str(reference),
        "gate": "FAIL" if fail else "PASS",
        "failures": fail,
        "errors": errors,
    }


class CudaEvents:
    def __init__(self):
        self.rt = ctypes.CDLL("/usr/local/cuda/lib64/libcudart.so")
        rt = self.rt
        rt.cudaEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        rt.cudaEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        rt.cudaEventSynchronize.argtypes = [ctypes.c_void_p]
        rt.cudaEventElapsedTime.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.a, self.b = ctypes.c_void_p(), ctypes.c_void_p()
        if rt.cudaEventCreate(ctypes.byref(self.a)) or rt.cudaEventCreate(
            ctypes.byref(self.b)
        ):
            raise RuntimeError("CUDA events")

    def start(self):
        if self.rt.cudaEventRecord(self.a, None):
            raise RuntimeError("CUDA start event")

    def stop(self):
        if self.rt.cudaEventRecord(self.b, None) or self.rt.cudaEventSynchronize(
            self.b
        ):
            raise RuntimeError("CUDA stop event")
        out = ctypes.c_float()
        if self.rt.cudaEventElapsedTime(ctypes.byref(out), self.a, self.b):
            raise RuntimeError("CUDA elapsed")
        return out.value


def interval_union_ms(rows):
    intervals = sorted((int(x["start_ns"]), int(x["end_ns"])) for x in rows)
    total = 0
    end = None
    for start, stop in intervals:
        if stop < start:
            raise AssertionError("negative activity duration")
        if end is None or start >= end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total / 1e6


def segment_trace(path, windows, representative_index=0, allow_preparation=False):
    """Assign by exact CUDA API correlation, without CPU/GPU clock assumptions."""
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    groups = [[] for _ in windows]
    prepared = [r for r in rows if int(r["step_id"]) >= 2**63]
    if prepared and not allow_preparation:
        raise AssertionError("Unexpected excluded preparation")
    rows = [r for r in rows if int(r["step_id"]) < 2**63]
    for row in rows:
        if row["kind"] not in ("kernel", "copy", "memset"):
            raise AssertionError(("unexpected activity", row))
        step = int(row["step_id"])
        correlation = int(row["correlation_id"])
        if not 1 <= step <= len(windows) or correlation <= 0:
            raise AssertionError(("unmarked or unknown step", row))
        if int(row["end_ns"]) < int(row["start_ns"]):
            raise AssertionError("negative activity duration")
        groups[step - 1].append(row)
    samples = []
    excursions = []
    for i, ((start, stop), selected) in enumerate(zip(windows, groups)):
        kernels = [x for x in selected if x["kind"] == "kernel"]
        if not kernels:
            raise AssertionError(("No kernels in correlated complete core step", i))
        names = [x["name"] for x in kernels]
        for x in selected:
            v = max(start - int(x["start_ns"]), int(x["end_ns"]) - stop, 0)
            if v:
                excursions.append(v)
        samples.append(
            {
                "step": i,
                "step_id": i + 1,
                "start_ns": start,
                "end_ns": stop,
                "kernel_sum_ms": sum(
                    (int(x["end_ns"]) - int(x["start_ns"])) / 1e6 for x in kernels
                ),
                "activity_sum_ms": sum(
                    (int(x["end_ns"]) - int(x["start_ns"])) / 1e6 for x in selected
                ),
                "activity_union_ms": interval_union_ms(selected),
                "kernel_count": len(kernels),
                "copy_count": sum(x["kind"] == "copy" for x in selected),
                "memset_count": sum(x["kind"] == "memset" for x in selected),
                "kernel_sequence_sha256": hashlib.sha256(
                    json.dumps(names).encode()
                ).hexdigest(),
            }
        )
    sequences = collections.Counter(x["kernel_sequence_sha256"] for x in samples)
    return samples, {
        "activity_assignment": "CUDA API callback correlation ID with process-global synchronous step marker; independent of CPU/GPU timestamp calibration",
        "unique_kernel_sequence_count": len(sequences),
        "sequence_occurrences": dict(sequences),
        "all_steps_have_same_kernel_sequence": len(sequences) == 1,
        "all_activity_assigned_exactly_once": True,
        "received_activity_count": len(rows) + len(prepared),
        "assigned_activity_count": sum(len(x) for x in groups) + len(prepared),
        "untimed_preparation_activity_count": len(prepared),
        "untimed_preparation_gpu_ms": sum(
            (int(r["end_ns"]) - int(r["start_ns"])) / 1e6 for r in prepared
        ),
        "cpu_gpu_timestamp_excursion_count": len(excursions),
        "maximum_cpu_gpu_timestamp_excursion_ns": max(excursions, default=0),
        "representative_step_activity": groups[representative_index],
    }


def measure(runtime, args, mode):
    import numpy as np

    sys.path.insert(0, str(ROOT / "benchmarks"))
    from support import Monitor, cool_down

    runtime.prepare_step(mode)
    result = runtime.run(mode)
    runtime.sync()
    del result
    runtime.collect()
    # Initialize every tracer/event object before the captured warmup. Warming
    # before CUPTI initialization let the GPU idle and change clocks mid-sample.
    tracer = ctypes.CDLL(str(resolve_path(args.tracer)))
    tracer.trace_start.argtypes = [ctypes.c_char_p]
    tracer.trace_start.restype = ctypes.c_int
    tracer.trace_stop.argtypes = []
    tracer.trace_stop.restype = ctypes.c_int
    tracer.trace_mark.argtypes = [ctypes.c_uint64]
    tracer.trace_mark.restype = None
    tracer.trace_unmarked.argtypes = []
    tracer.trace_unmarked.restype = ctypes.c_uint64
    tracer.trace_timestamp.argtypes = []
    tracer.trace_timestamp.restype = ctypes.c_uint64
    event = CudaEvents()
    path = (
        args.output_dir / f"full-{args.case}-{args.framework}-{args.label}-{mode}.tsv"
    )
    cold = cool_down(limit=args.cooling_limit)
    monitor = Monitor()
    if tracer.trace_start(str(path).encode()):
        raise RuntimeError("CUPTI start failed")
    warm_windows = []
    warm_start = time.perf_counter()
    count = 0
    try:
        while count < 10 or time.perf_counter() - warm_start < args.warm_seconds:
            if mode == "backward":
                tracer.trace_mark(2**63 + count + 1)
            runtime.prepare_step(mode)
            tracer.trace_mark(count + 1)
            begin = int(tracer.trace_timestamp())
            result = runtime.run(mode)
            runtime.sync()
            del result
            end = int(tracer.trace_timestamp())
            tracer.trace_mark(0)
            warm_windows.append((begin, end))
            count += 1
    except BaseException:
        tracer.trace_stop()
        monitor.finish(warm_start, time.perf_counter())
        raise
    windows = []
    walls = []
    events = []
    start = time.perf_counter()
    try:
        for sample_index in range(args.reps):
            if mode == "backward":
                tracer.trace_mark(2**63 + count + sample_index + 1)
            runtime.prepare_step(mode)
            t = time.perf_counter()
            tracer.trace_mark(count + sample_index + 1)
            begin = int(tracer.trace_timestamp())
            event.start()
            if begin <= 0:
                raise RuntimeError("Invalid CUPTI timestamp")
            result = runtime.run(mode)
            runtime.sync()
            event_ms = event.stop()
            del result
            end = int(tracer.trace_timestamp())
            tracer.trace_mark(0)
            if end < begin:
                raise RuntimeError("Non-monotonic CUPTI timestamp")
            windows.append((begin, end))
            events.append(event_ms)
            walls.append((time.perf_counter() - t) * 1000)
        finish = time.perf_counter()
    finally:
        status = tracer.trace_stop()
        hardware = monitor.finish(start, locals().get("finish", time.perf_counter()))
    if status:
        raise RuntimeError("CUPTI stop failed")
    tracer.trace_dropped.restype = ctypes.c_ulonglong
    tracer.trace_errors.restype = ctypes.c_uint
    trace_integrity = {
        "dropped_records": int(tracer.trace_dropped()),
        "errors": int(tracer.trace_errors()),
        "unmarked_activity": int(tracer.trace_unmarked()),
    }
    if any(trace_integrity.values()):
        raise RuntimeError("CUPTI incomplete trace: " + str(trace_integrity))
    all_samples, segmentation = segment_trace(
        path,
        warm_windows + windows,
        representative_index=count,
        allow_preparation=mode == "backward",
    )
    samples = all_samples[count:]
    formal_sequences = collections.Counter(x["kernel_sequence_sha256"] for x in samples)
    segmentation.update(
        {
            "captured_warmup_steps": count,
            "formal_steps": len(samples),
            "all_formal_steps_have_same_kernel_sequence": len(formal_sequences) == 1,
            "formal_sequence_occurrences": dict(formal_sequences),
            "warmup_windows": warm_windows,
            "warmup_policy": "At least 2 seconds after tracer/event setup and trace_start, explicitly timestamped and uniquely assigned outside formal 50 samples.",
        }
    )
    primary = [x["activity_sum_ms"] for x in samples]
    kernel = [x["kernel_sum_ms"] for x in samples]
    counts = collections.Counter(x["kernel_count"] for x in samples)
    report = {
        "mode": mode,
        "trace_path": str(path),
        "cooling_limit_c": args.cooling_limit,
        "warmup_seconds": args.warm_seconds,
        "primary_metric": "all core GPU activity sum per timestamp-delimited step; no subtraction",
        "median_group_ms": float(np.median(primary)),
        "kernel_sum_ms": float(np.median(kernel)),
        "group_p10_ms": float(np.quantile(primary, 0.1)),
        "group_p90_ms": float(np.quantile(primary, 0.9)),
        "event_ms": float(np.median(events)),
        "wall_ms": float(np.median(walls)),
        "wall_p10_ms": float(np.quantile(walls, 0.1)),
        "wall_p90_ms": float(np.quantile(walls, 0.9)),
        "wall_total_per_step_ms": (finish - start) * 1000 / args.reps,
        "reps": args.reps,
        "warm_steps": count,
        "hardware": hardware,
        "cold": cold,
        "trace_integrity": trace_integrity,
        "samples": samples,
        "segmentation": segmentation,
        "kernel_count_distribution": dict(counts),
        "event_samples_ms": events,
        "wall_samples_ms": walls,
    }
    if hardware["gate"] != "PASS":
        raise RuntimeError("Core hardware gate failed: " + str(hardware))
    return report


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--framework", choices=["prepare", "torch", "jt", "compare"])
    p.add_argument(
        "--case", choices=["dense", "production", "long"], default="production"
    )
    p.add_argument("--label", default="control")
    p.add_argument("--source-root", type=resolve_path, default=ROOT)
    p.add_argument(
        "--baseline-root",
        type=resolve_path,
        default=Path(os.environ.get("CYFA_BASELINE_ROOT", "baseline")),
    )
    p.add_argument(
        "--output-dir", type=resolve_path, default=ROOT / "results/benchmark/core"
    )
    p.add_argument("--original-fixture", type=resolve_path)
    p.add_argument("--prepared", type=resolve_path)
    p.add_argument("--reference", type=resolve_path)
    p.add_argument("--actual", type=resolve_path)
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--warm-seconds", type=float, default=2.0)
    p.add_argument("--cooling-limit", type=int, default=45)
    p.add_argument("--modes", default="forward,forward_backward")
    p.add_argument(
        "--tracer",
        type=resolve_path,
        default=ROOT / "results/benchmark/native/activity.so",
    )
    p.add_argument(
        "--check-source",
        action="store_true",
        help="CPU-only AST extraction/compile/audit; no framework import.",
    )
    return p.parse_args()


def main():
    args = parse()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adapter, audit = build_adapter(args.baseline_root, args.output_dir)
    if args.check_source:
        print(json.dumps({"cpu_only": True, "adapter_audit": audit}), flush=True)
        return
    if not args.framework:
        raise ValueError("--framework required unless --check-source")
    if args.framework == "compare":
        if not args.actual or not args.reference:
            raise ValueError("compare requires --actual and --reference")
        result = compare_values(args.actual, args.reference)
        write_json(
            args.output_dir / f"comparison-{args.case}-{args.label}.json", result
        )
        print(json.dumps(result), flush=True)
        raise SystemExit(0 if result["gate"] == "PASS" else 1)
    for key, path_value in {
        "JITTOR_HOME": ROOT / ".cache",
        "TMPDIR": ROOT / ".cache/tmp",
        "TRITON_CACHE_DIR": ROOT / ".cache/triton-baseline",
        "TORCH_EXTENSIONS_DIR": ROOT / ".cache/torch-extensions",
        "XDG_CACHE_HOME": ROOT / ".cache/xdg",
        "CUDA_CACHE_PATH": ROOT / ".cache/cuda",
    }.items():
        target = resolve_path(os.environ.get(key, path_value))
        target.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(target)
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError(
            "Set CUDA_VISIBLE_DEVICES to the physical preparation/benchmark GPU before launch."
        )
    if args.framework == "prepare":
        prepare(args)
        return
    if args.reps < 50 or args.warm_seconds < 2:
        raise ValueError(
            "Formal core sampling requires >=50 reps and >=2 seconds warmup"
        )
    os.environ["CYFA_CUDA_PROFILE"] = "0"
    import numpy as np

    path = resolve_path(args.prepared or args.output_dir / f"prepared-{args.case}.npz")
    with np.load(path) as z:
        arrays = {k: np.ascontiguousarray(z[k]) for k in KEYS + ("do",)}
    B, T, H, D = arrays["q"].shape
    expected = {
        "dense": (4, 2048, 4, 256),
        "production": (32, 2048, 4, 256),
        "long": (4, 8192, 4, 256),
    }
    if (B, T, H, D) != expected[args.case] or arrays["table"].shape != (H, 127, 128):
        raise AssertionError("prepared shape mismatch")
    visible = os.environ["CUDA_VISIBLE_DEVICES"]
    if "," in visible or not visible.strip():
        raise ValueError("Exactly one physical visible GPU is required")
    runtime = (
        TorchRuntime(args, arrays, adapter, audit)
        if args.framework == "torch"
        else JittorRuntime(args, arrays)
    )
    prepared_tensor_shapes = {k: list(v.shape) for k, v in arrays.items()}
    del arrays
    report = {
        "contract": CONTRACT,
        "framework": args.framework,
        "case": args.case,
        "label": args.label,
        "scope": "Complete prepared-input CyFA core; no q/k RMSNorm, basis/table construction, input/output layout, model or optimizer.",
        "prepared_path": str(path),
        "prepared_sha256": sha(path),
        "shape": [B, T, H, D, 128],
        "shape_axes": ["B", "T", "H", "Dk", "M"],
        "prepared_tensor_shapes": prepared_tensor_shapes,
        "mode": "multiple",
        "modes": {},
        "harness_path": str(Path(__file__).resolve()),
        "harness_sha256": sha(__file__),
        "tracer_path": str(args.tracer),
        "tracer_sha256": sha(args.tracer),
        "telemetry_source_sha256": sha(ROOT / "benchmarks/support.py"),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "provenance": runtime.provenance(),
        "adapters_sha256": audit["adapter_sha256"],
        "numerical": export_validation(runtime, args),
    }
    report["objective_adapter"] = {
        "version": "jittor_unit_cotangent_v1"
        if args.framework == "jt"
        else "unchanged_torch_explicit_bf16_grad_output",
        "unit_seed_only": True,
        "preprepared_cotangent_dtype": "BF16",
        "protocol_version": "explicit_bf16_vjp_trace4_launch_correlation_repeat_all_v1",
        "vjp_harness_path": str(Path(__file__).resolve()),
        "vjp_harness_sha256": sha(__file__),
        "cotangent_dtype": "BF16",
    }
    report["source_sha256"] = report["provenance"]["source_sha256"]
    report["adapter_sha256"] = audit["adapter_sha256"]
    output = args.output_dir / f"full-{args.case}-{args.framework}-{args.label}.json"
    write_json(output, report)
    for mode in (m.strip() for m in args.modes.split(",")):
        if mode not in ("forward", "forward_backward", "inference", "backward"):
            raise ValueError("unsupported mode " + mode)
        print("MEASURING", args.framework, args.case, mode, flush=True)
        report["modes"][mode] = measure(runtime, args, mode)
        mode_report = {k: v for k, v in report.items() if k not in ("modes", "mode")}
        mode_report.update(report["modes"][mode])
        write_json(
            args.output_dir
            / f"full-{args.case}-{args.framework}-{args.label}-{mode}.json",
            mode_report,
        )
        write_json(output, report)
        print("RESULT", mode, report["modes"][mode]["median_group_ms"], flush=True)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
