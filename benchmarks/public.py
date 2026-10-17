"""Full public CyFA operator benchmark, independently gated from prepared core."""

import argparse
import contextlib
import os
import sys
import time
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "benchmarks"))
sys.path.insert(0, str(R))
import core as common

KEYS = (
    "q",
    "k",
    "v",
    "g",
    "delta",
    "beta",
    "readout",
    "q_norm_weight",
    "k_norm_weight",
)
common.KEYS = KEYS
p = argparse.ArgumentParser()
p.add_argument("--framework", choices=["jt", "torch"], required=True)
p.add_argument("--case", choices=["dense", "production", "long"], required=True)
p.add_argument("--label", required=True)
p.add_argument(
    "--baseline-root",
    type=common.resolve_path,
    default=Path(os.environ.get("CYFA_BASELINE_ROOT", "baseline")),
)
p.add_argument("--reference", type=common.resolve_path)
a = p.parse_args()
a.output_dir = R / "results/benchmark/public"
a.output_dir.mkdir(parents=True, exist_ok=True)
a.source_root = R
a.tracer = R / "results/benchmark/native/activity.so"
a.reps = 50
a.warm_seconds = 2.0
a.cooling_limit = 50
assert (
    os.environ.get("CUDA_VISIBLE_DEVICES")
    and "," not in os.environ["CUDA_VISIBLE_DEVICES"]
), "Select exactly one GPU"
os.environ["CYFA_CUDA_PROFILE"] = "0"
fixture = R / f"results/benchmark/fixture-{a.case}.npz"
with np.load(fixture) as z:
    arrays = {k: np.ascontiguousarray(z[k]) for k in KEYS + ("do",)}


class JittorRuntime:
    def __init__(self, arrays):
        import jittor as jt

        import cyclic_flow_attention
        from cyclic_flow_attention.ops import chunk_cyfa

        jt.flags.use_cuda = 1
        jt.flags.use_tensorcore = 0
        self.jt = jt
        self.fn = chunk_cyfa
        self.path = Path(cyclic_flow_attention.__file__).resolve().parent
        self.x = {
            k: jt.array(arrays[k])
            .cast("bfloat16" if k in ("q", "k", "v") else "float32")
            .start_grad()
            for k in KEYS
        }
        self.do = jt.array(arrays["do"]).bfloat16().stop_grad()

        class UnitCotangent(jt.Function):
            def execute(self, y, do):
                self.do = do
                return jt.zeros((1,), dtype="float32")

            def grad(self, seed):
                return self.do, None

        self.vjp = UnitCotangent.apply
        self.sync()

    def sync(self):
        self.jt.sync_all(True)

    def prepare_step(self, mode):
        pass

    def run(self, mode):
        with self.jt.no_grad() if mode == "inference" else contextlib.nullcontext():
            y, states = self.fn(
                **self.x, output_final_state=True, q_norm_eps=1e-6, k_norm_eps=1e-6
            )
        if mode == "forward_backward":
            return y, states, self.jt.grad(self.vjp(y, self.do), list(self.x.values()))
        return y, states

    def array(self, v):
        return v.float32().numpy()

    def canonical(self, v, key):
        return self.array(v)

    def collect(self):
        self.jt.gc()

    def provenance(self):
        return {
            "framework_version": self.jt.__version__,
            "runtime_imports": {
                "torch": "torch" in sys.modules,
                "triton": "triton" in sys.modules,
            },
            "package_path": str(self.path),
            "leaf_names": list(KEYS),
            "source_sha256": {
                str(p): common.sha(p) for p in sorted(self.path.rglob("*.py"))
            },
        }


class TorchRuntime:
    def __init__(self, arrays):
        os.environ["CYFA_BASELINE_ROOT"] = str(a.baseline_root)
        sys.path.insert(0, str(a.baseline_root))
        import torch
        import triton
        from support import load_baseline

        chunk_cyfa = load_baseline()
        assert torch.__version__ == "2.6.0+cu124" and triton.__version__ == "3.2.0"
        self.torch = torch
        self.triton = triton
        self.fn = chunk_cyfa
        self.x = {
            k: torch.tensor(
                arrays[k],
                device="cuda",
                dtype=torch.bfloat16 if k in ("q", "k", "v") else torch.float32,
                requires_grad=True,
            )
            for k in KEYS
        }
        self.do = torch.tensor(arrays["do"], device="cuda", dtype=torch.bfloat16)
        self.sync()

    def sync(self):
        self.torch.cuda.synchronize()

    def prepare_step(self, mode):
        pass

    def run(self, mode):
        with self.torch.no_grad() if mode == "inference" else contextlib.nullcontext():
            y, states = self.fn(
                **self.x, output_final_state=True, q_norm_eps=1e-6, k_norm_eps=1e-6
            )
        if mode == "forward_backward":
            return (
                y,
                states,
                self.torch.autograd.grad(y, list(self.x.values()), self.do),
            )
        return y, states

    def array(self, v):
        return v.detach().float().cpu().numpy()

    def canonical(self, v, key):
        return self.array(v)

    def collect(self):
        self.torch.cuda.empty_cache()

    def provenance(self):
        return {
            "framework_version": self.torch.__version__,
            "triton": self.triton.__version__,
            "baseline_root": str(a.baseline_root),
            "leaf_names": list(KEYS),
            "public_function_module": self.fn.__module__,
            "source_sha256": {
                str(p): common.sha(p)
                for p in [
                    a.baseline_root / x
                    for x in (
                        "fla/ops/cyfa/chunk.py",
                        "fla/ops/cyfa/qk_norm.py",
                        "fla/ops/cyfa/utils.py",
                        "fla/ops/common/chunk_h.py",
                        "fla/ops/common/chunk_o.py",
                    )
                ]
                + [R / "benchmarks/support.py"]
            },
        }


runtime = JittorRuntime(arrays) if a.framework == "jt" else TorchRuntime(arrays)
shapes = {k: list(v.shape) for k, v in arrays.items()}
del arrays
report = {
    "contract": "public_cyfa_operator_v3_explicit_bf16_vjp",
    "scope": "Full public chunk_cyfa: q/k normalization and its weight gradients, all layouts, basis/table construction, complete CyFA core, output and all final states. No model, projection, optimizer or training-loop credit. Core performance must independently pass.",
    "framework": a.framework,
    "case": a.case,
    "label": a.label,
    "fixture_path": str(fixture),
    "fixture_sha256": common.sha(fixture),
    "input_shapes": shapes,
    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    "harness_path": str(Path(__file__).resolve()),
    "harness_sha256": common.sha(__file__),
    "shared_sampler_sha256": common.sha(R / "benchmarks/core.py"),
    "tracer_sha256": common.sha(a.tracer),
    "telemetry_sha256": common.sha(R / "benchmarks/support.py"),
    "provenance": runtime.provenance(),
    "numerical": common.export_validation(runtime, a),
    "modes": {},
}
report["numerical"]["repeat_scope"] = (
    "public out, three final states and all nine public input gradients; unit BF16 output cotangent; final-state cotangents zero"
)
out = a.output_dir / f"api-{a.case}-{a.framework}-{a.label}.json"
common.write_json(out, report)


def measure_wall(mode):
    from support import Monitor, cool_down

    result = runtime.run(mode)
    runtime.sync()
    del result
    events = common.CudaEvents()
    cold = cool_down(50)
    monitor = Monitor()
    start = time.perf_counter()
    n = 0
    while n < 10 or time.perf_counter() - start < 2.0:
        result = runtime.run(mode)
        runtime.sync()
        del result
        n += 1
    walls = []
    gpu = []
    begin = time.perf_counter()
    for _ in range(50):
        t = time.perf_counter()
        events.start()
        result = runtime.run(mode)
        runtime.sync()
        gpu.append(events.stop())
        del result
        walls.append((time.perf_counter() - t) * 1000)
    end = time.perf_counter()
    hardware = monitor.finish(begin, end)
    assert hardware["gate"] == "PASS"
    return {
        "protocol": "unprofiled_synchronized_public_operator_v1",
        "primary_metric": "median synchronized wall time including output release, without CUPTI initialization or callbacks",
        "wall_ms": float(np.median(walls)),
        "wall_p10_ms": float(np.quantile(walls, 0.1)),
        "wall_p90_ms": float(np.quantile(walls, 0.9)),
        "wall_samples_ms": walls,
        "event_samples_ms": gpu,
        "reps": 50,
        "warmup_seconds": 2.0,
        "warm_steps": n,
        "cooling_limit_c": 50,
        "cold": cold,
        "hardware": hardware,
    }


# All primary wall measurements precede the first CUPTI initialization.
report["unprofiled_wall"] = {}
for mode in ("forward", "forward_backward", "inference"):
    report["unprofiled_wall"][mode] = measure_wall(mode)
    common.write_json(out, report)
    print(
        "API_WALL", a.case, mode, report["unprofiled_wall"][mode]["wall_ms"], flush=True
    )

for mode in ("forward", "forward_backward", "inference"):
    # Each execution mode is independently evaluated before measurement.
    values = []
    for _ in range(3):
        result = runtime.run(mode)
        runtime.sync()
        v = {"out": runtime.array(result[0])}
        v.update({"state" + str(i): runtime.array(s) for i, s in enumerate(result[1])})
        values.append(v)
        del result
    m = common.measure(runtime, a, mode)
    m["primary_metric"] = (
        "Profiled full public operator GPU activity; synchronized primary wall time is measured separately before CUPTI initialization"
    )
    m["mode_outputs_finite"] = all(
        np.isfinite(x).all() for v in values for x in v.values()
    )
    m["mode_repeat_max_abs"] = {
        k: max(float(np.max(np.abs(v[k] - values[0][k]))) for v in values[1:])
        for k in values[0]
    }
    assert m["mode_outputs_finite"] and not any(m["mode_repeat_max_abs"].values())
    if a.reference:
        with np.load(a.reference) as z:
            m["mode_oracle_rrms"] = {
                k: float(
                    np.sqrt(np.mean((x.astype("float64") - z[k]) ** 2))
                    / max(np.sqrt(np.mean(z[k].astype("float64") ** 2)), 1e-12)
                )
                for k, x in values[0].items()
            }
        assert all(v < 0.005 for v in m["mode_oracle_rrms"].values())
    report["modes"][mode] = m
    common.write_json(out, report)
    print(
        "API_RESULT",
        a.case,
        mode,
        "wall",
        m["wall_ms"],
        "GPU",
        m["median_group_ms"],
        flush=True,
    )
