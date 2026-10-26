# Hy4-preview RMS + dynamic INT8 (experimental direct API)

`rms_scaled_int8_quant(x, weight, eps=1e-5)` returns the BF16 normalized
activation, signed INT8 values, and an FP32 per-row scale. Keeping the BF16
activation preserves the separate BF16 indexer consumer. The operator has no
global ATen registration and does not change the INT8 GEMM provider or weights.
The caller opts in explicitly and owns its original path when metadata declines.

Inputs are CUDA BF16 `[M,N]`, contiguous on the last axis with nonoverlapping
rows, and a contiguous same-device BF16 weight `[N]`. Grad must be disabled.
Only N1..4096 is supported. Optional norm/out/scale buffers must have disjoint
storage and validated dtype/shape/strides. Unsupported metadata raises
`NotImplementedError`; there is no Torch compute fallback.

The reduction uses FP32, casts to BF16 before multiplying by the norm weight,
and casts again before quantizing. INT8 uses the materialized BF16 value,
round-to-nearest division and saturating round-to-nearest-even conversion.
Zero rows produce zero scale and quantized values. Nonfinite, training and
unmeasured shape numerics are outside the qualified model domain.

## Existing evidence and limitations

On a pinned H100/SM90 runtime (Torch2.13+cu129, FlagTree3.6), existing complete
RMS→INT8 q_b + BF16 indexer cold graph medians were 38.112→33.312µs at M208 and
37.600→32.896µs at M256. These are the archived R1 fused-producer comparisons,
not measurements of this PR on current main. The module excludes cache, TopK,
MLA, MoE and TP communication. The official BF16 q_b path has a different
precision contract; its cost is not a framework-only comparison.

The unchanged implementation also passed a standard world1 actual
Indexer.prepare_inputs + RMS/ColumnParallel compressed-tensors INT8 integration:
M208/M256, independent CPU norm/quant/projection references, OFF/ON byte equality,
eager/ordinary graph equality and unchanged parameter/loader names. GPU job
c51e0835 exited0. It did not load a checkpoint, benchmark serving, or qualify
TP16/EP16, request reuse or real-weight accuracy.

No autotune is added: this is an explicit fixed-schedule exception preserving
the measured reduction and BF16 rounding (next_power_of_2(N), warps4, stages1).
New scheduling must be separately qualified. Direct exports are used because
there is no single matching ATen schema; no global dispatcher entry is changed.

The added functional and standalone benchmark entrypoints are:

```bash
python -m pytest -q tests/test_rms_scaled_int8_quant.py
python -m pytest -q benchmark/test_rms_scaled_int8_quant_perf.py --level core
```

The standalone Torch semantic baseline has a different launch/implementation
cost and must not be substituted for the archived production-consumer baseline.
New GPU tests and current-main performance remain pending; keep this proposal
draft until those checks and model integration are complete. The source class
names used by the model and loaders remain unchanged.
