# Operator benchmarks

Use separate Jittor and PyTorch/Triton environments. The performance reference is PyTorch 2.6.0+cu124 and **Triton 3.2.0**. Runtime code does not import either package.

```bash
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.run \
  --jittor-python /path/to/jittor-env/bin/python \
  --baseline-python /path/to/fla-env/bin/python \
  --baseline-root /path/to/flash-linear-attention-cyfa \
  --patched-baseline-root /path/to/optional-sm80-tuned-checkout \
  --gpu 0 --cases dense,long,production --scope all
```

Use a checkout of the [PyTorch implementation](https://github.com/Chyxx/CyclicFlowAttention) as the baseline. The optional SM80 patch is included as [`sm80.patch`](sm80.patch); apply it to a separate checkout with `git apply /path/to/CyFA-Jittor/benchmarks/sm80.patch`. The runner records each baseline revision and diff without editing the checkout. `support.py` adapts optional-package discovery and the unsupported `launch_pdl=False` keyword for Triton 3.2 without changing kernels.

A CUDA toolkit with CUPTI headers and libraries, `g++`, and `nvidia-smi` are required. The runner compiles the single CUPTI tracing helper into `results/benchmark/native`. Generated binaries, common inputs, raw numerical exports and traces go to ignored paths. Each shape/backend/protocol gets a fresh Python process, preventing cross-shape autotune reuse. Run on an otherwise idle GPU; foreign processes, dropped records, unmarked work and thermal throttling fail the measurement.

## Boundaries

- **Prepared core:** identical BF16 normalized q/k/v, FP32 gates/readout table and BF16 output cotangent. All preparation occurs before timing. Both scalar recurrences, readout/softmax, clock/phase, final states, recomputation and complete backward work remain included. The baseline's custom autograd class is copied verbatim into a private namespace; only normalization becomes an identity on prepared inputs and final-state transpose copies become views. The original files are never modified.
- **Public operator:** the unmodified public `chunk_cyfa` boundary, including normalization/gradients, readout table construction and layouts. Its unprofiled synchronized wall time is measured before CUPTI initializes; GPU activity is then measured independently.

Forward runs with autograd enabled. `forward_backward` includes an actual forward and all leaf gradients, retaining output/final states through synchronization. `backward` performs and synchronizes a fresh forward **before** each timer, then measures only its backward work; preparatory launches have separate trace IDs and remain auditable in the trace. It is not calculated by subtracting medians. `inference` uses no-grad execution.

The primary GPU metric sums all correlated kernel/copy/memset durations for each complete call. There are 50 formal samples after at least two seconds of warmup. These are per-call measurements, not isolated best-kernel times. Ordinary frequency differences are retained in the evidence. Core and public-operator results are reported separately from model and training-loop performance.

## Correctness and memory

Every backend exports the output, three final states and all seven prepared-core (nine public-operator) gradients. All must be finite. Relative RMS error ceilings are 0.005 for outputs/states, 0.01 for ordinary gradients, and 0.02 for g/delta/beta gradients. Three repeats require relative RMS at most 1e-6 and max-error/reference-max at most 1e-5; bitwise equality is recorded separately. Jittor uses a unit-cotangent adapter to supply the same BF16 output gradient directly, without a generic reduction scaffold; this adapter is confined to benchmarks.

The native implementation saves two BF16 state histories; the original baseline recomputes them. FP32 accumulators and persistent states are preserved. Account for these different activation-memory costs when comparing latency.
