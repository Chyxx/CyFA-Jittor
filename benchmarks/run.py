"""Run fresh-process core and public-operator comparisons against Triton 3.2."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jittor-python", default=sys.executable)
    parser.add_argument("--baseline-python", required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument(
        "--patched-baseline-root", help="Optional existing SM80-tuned baseline checkout"
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--cases", default="dense,long,production")
    parser.add_argument("--scope", choices=["core", "public", "all"], default="all")
    args = parser.parse_args()
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=args.gpu, PYTHONPATH=str(ROOT))
    output = ROOT / "results/benchmark"
    output.mkdir(parents=True, exist_ok=True)
    native = output / "native"
    native.mkdir(parents=True, exist_ok=True)
    cuda = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    for name in ("activity",):
        subprocess.run(
            [
                "g++",
                "-O3",
                "-std=c++17",
                "-shared",
                "-fPIC",
                str(ROOT / "benchmarks/trace.cpp"),
                "-I" + str(cuda / "include"),
                "-I" + str(cuda / "extras/CUPTI/include"),
                "-L" + str(cuda / "lib64"),
                "-L" + str(cuda / "extras/CUPTI/lib64"),
                "-Wl,-rpath," + str(cuda / "extras/CUPTI/lib64"),
                "-lcupti",
                "-lcuda",
                "-ldl",
                "-o",
                str(native / (name + ".so")),
            ],
            check=True,
        )
    branches = [("original", args.baseline_root)]
    if args.patched_baseline_root:
        branches.append(("patched", args.patched_baseline_root))
    branches.append(("jt", args.baseline_root))
    for case in args.cases.split(","):
        if not (output / f"fixture-{case}.npz").exists():
            subprocess.run(
                [
                    args.baseline_python,
                    "-c",
                    "from benchmarks.run import make_fixture; make_fixture("
                    + repr(case)
                    + ")",
                ],
                cwd=ROOT,
                env=environment,
                check=True,
            )
        for role, baseline in branches:
            environment["CYFA_BASELINE_ROOT"] = str(Path(baseline).resolve())
            python = args.jittor_python if role == "jt" else args.baseline_python
            framework = "jt" if role == "jt" else "torch"
            common = [
                "--framework",
                framework,
                "--case",
                case,
                "--label",
                role,
                "--baseline-root",
                baseline,
            ]
            core = output / "core"
            if args.scope in ("core", "all"):
                if not (core / f"prepared-{case}.npz").exists():
                    subprocess.run(
                        [
                            args.baseline_python,
                            "benchmarks/core.py",
                            "--framework",
                            "prepare",
                            "--case",
                            case,
                            "--baseline-root",
                            args.baseline_root,
                        ],
                        cwd=ROOT,
                        env=environment,
                        check=True,
                    )
            if args.scope in ("core", "all"):
                command = [
                    python,
                    "benchmarks/core.py",
                    *common,
                    "--cooling-limit",
                    "50",
                    "--modes",
                    "forward,backward,forward_backward,inference",
                ]
                if role == "jt":
                    command += [
                        "--reference",
                        str(core / f"full-{case}-torch-original-values.npz"),
                    ]
                subprocess.run(command, cwd=ROOT, env=environment, check=True)
            if args.scope in ("public", "all"):
                command = [python, "benchmarks/public.py", *common]
                if role == "jt":
                    command += [
                        "--reference",
                        str(
                            output / "public" / f"full-{case}-torch-original-values.npz"
                        ),
                    ]
                subprocess.run(command, cwd=ROOT, env=environment, check=True)


def make_fixture(case):
    import numpy as np
    import torch

    b, t = {"dense": (4, 2048), "long": (4, 8192), "production": (32, 2048)}[case]
    h, d, m = 4, 256, 128
    torch.manual_seed(42)
    values = {
        k: torch.randn(b, t, h, d, device="cuda", dtype=torch.bfloat16)
        for k in ("q", "k", "v")
    }
    values.update(
        g=-0.1 * torch.rand(b, t, h, device="cuda"),
        delta=torch.randn(b, t, h, device="cuda").sigmoid(),
        beta=torch.randn(b, t, h, device="cuda").sigmoid(),
        readout=torch.eye(m - 1, device="cuda").expand(h, -1, -1).clone()
        + 0.05 * torch.randn(h, m - 1, m - 1, device="cuda"),
        q_norm_weight=1 + 0.1 * torch.randn(d, device="cuda"),
        k_norm_weight=1 + 0.1 * torch.randn(d, device="cuda"),
    )
    values["do"] = torch.randn_like(values["v"])
    path = (
        Path(__file__).resolve().parents[1]
        / "results/benchmark"
        / f"fixture-{case}.npz"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{k: v.float().cpu().numpy() for k, v in values.items()})


if __name__ == "__main__":
    main()
