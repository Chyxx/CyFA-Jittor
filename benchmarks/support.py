"""Read-only GPU telemetry and passive cooling for comparable measurements."""

import csv
import datetime
import os
import subprocess
import threading
import time


def gpu_id():
    return os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]


def snapshot():
    fields = "uuid,name,clocks.sm,temperature.gpu,power.draw,clocks_throttle_reasons.sw_thermal_slowdown,clocks_throttle_reasons.hw_thermal_slowdown"
    row = next(
        csv.reader(
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--id=" + gpu_id(),
                    "--query-gpu=" + fields,
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).splitlines()
        )
    )
    v = [x.strip() for x in row]
    return {
        "physical_gpu": gpu_id(),
        "uuid": v[0],
        "name": v[1],
        "sm_mhz": int(v[2]),
        "temperature_c": int(v[3]),
        "power_w": float(v[4]),
        "sw_thermal": v[5] == "Active",
        "hw_thermal": v[6] == "Active",
    }


def other_processes():
    uuid = snapshot()["uuid"]
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    return [
        int(row.split(",")[1])
        for row in rows
        if row.split(",")[0].strip() == uuid and int(row.split(",")[1]) != os.getpid()
    ]


def cool_down(limit=45):
    end = time.monotonic() + 900
    last_print = 0
    while True:
        s = snapshot()
        others = other_processes()
        if others:
            raise RuntimeError("GPU contention: other process IDs " + str(others))
        if s["temperature_c"] <= limit and not s["sw_thermal"] and not s["hw_thermal"]:
            return s
        if time.monotonic() > end:
            raise TimeoutError("GPU did not cool to the comparison temperature")
        if time.monotonic() - last_print > 30:
            print("passive cooling", s, flush=True)
            last_print = time.monotonic()
        time.sleep(5)


class Monitor:
    def __init__(self):
        self.records = []
        self.foreign_start = other_processes()
        self.wall_anchor = time.time()
        self.perf_anchor = time.perf_counter()
        fields = "timestamp,clocks.sm,temperature.gpu,power.draw,clocks_throttle_reasons.sw_thermal_slowdown,clocks_throttle_reasons.hw_thermal_slowdown"
        self.proc = subprocess.Popen(
            [
                "nvidia-smi",
                "--id=" + gpu_id(),
                "--query-gpu=" + fields,
                "--format=csv,noheader,nounits",
                "--loop-ms=50",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self.thread = threading.Thread(target=self.collect, daemon=True)
        self.thread.start()

    def collect(self):
        for line in self.proc.stdout:
            try:
                v = [x.strip() for x in line.split(",")]
                sample_wall = datetime.datetime.strptime(
                    v[0], "%Y/%m/%d %H:%M:%S.%f"
                ).timestamp()
                self.records.append(
                    {
                        "at": self.perf_anchor + sample_wall - self.wall_anchor,
                        "driver_timestamp": v[0],
                        "received_at": time.perf_counter(),
                        "sm_mhz": int(v[1]),
                        "temperature_c": int(v[2]),
                        "power_w": float(v[3]),
                        "sw_thermal": v[4] == "Active",
                        "hw_thermal": v[5] == "Active",
                    }
                )
            except (ValueError, IndexError):
                pass

    def finish(self, start, end):
        self.proc.terminate()
        self.proc.wait(timeout=5)
        self.thread.join(timeout=1)
        selected = [x for x in self.records if start - 0.1 <= x["at"] <= end + 0.02]
        # Very short decode runs can be shorter than a single telemetry interval.
        if not selected and self.records:
            selected = self.records[-2:]
        warm = [x for x in self.records if x["at"] >= start - 0.5]
        checked = warm or selected
        foreign_end = other_processes()
        valid = (
            not self.foreign_start
            and not foreign_end
            and bool(selected)
            and all(
                not x["sw_thermal"] and not x["hw_thermal"] and x["temperature_c"] <= 75
                for x in checked
            )
        )
        return {
            "timestamp_source": "nvidia-smi driver query timestamp; independent of Python/GIL arrival",
            "foreign_pids_before": self.foreign_start,
            "foreign_pids_after": foreign_end,
            "physical_gpu": gpu_id(),
            "gate": "PASS" if valid else "FAIL",
            "maximum_temperature_c": max(
                (x["temperature_c"] for x in checked), default=None
            ),
            "minimum_sm_mhz": min((x["sm_mhz"] for x in selected), default=None),
            "maximum_sm_mhz": max((x["sm_mhz"] for x in selected), default=None),
            "telemetry_samples": len(selected),
            "records": selected,
        }


def load_baseline():
    import importlib.util
    import os
    import sys
    import types

    _find_spec = importlib.util.find_spec

    def _optional_spec(name, package=None):
        try:
            return _find_spec(name, package)
        except ModuleNotFoundError:
            return None

    importlib.util.find_spec = _optional_spec
    import fla  # noqa: F401 -- initialize FLA before installing its ops namespace
    import triton.runtime.jit

    _original_run = triton.runtime.jit.JITFunction.run

    def _run_32(self, *args, **kwargs):
        pdl = kwargs.pop("launch_pdl", False)
        if pdl:
            raise RuntimeError("PDL is unavailable on the A100 baseline")
        return _original_run(self, *args, **kwargs)

    triton.runtime.jit.JITFunction.run = _run_32
    # Import only the required operator family, avoiding unrelated optional models.
    ops = types.ModuleType("fla.ops")
    ops.__path__ = [os.environ["CYFA_BASELINE_ROOT"] + "/fla/ops"]
    sys.modules["fla.ops"] = ops
    from fla.ops.cyfa import chunk_cyfa

    importlib.util.find_spec = _find_spec
    return chunk_cyfa
