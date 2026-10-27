# Hy4-preview E23 operator delivery

These four NVIDIA CUDA direct APIs recover numerical implementations used by the archived
E23 Hy4-preview source bundle. They are explicit calls; importing this package
does not enable model scheduling or install a new aten registration. Other
backends expose an explicit unsupported error without importing NVIDIA tuning
configuration. The RMSNorm
dual-output API is tracked separately in PR #1001.

## Frozen project and validation contract

| Field | Contract |
| --- | --- |
| Base | FlagGems-vllm main d24cf945ed4fc2fd5b52d9c019026a809a96577a |
| Editable | Three ops, their exports/catalog, tests, benchmarks and this document |
| Read-only | Archived E23 numerical sources and historical GPU reports |
| Build | Pure Python/Triton files; installed repository environment required |
| Validation | Per-op pytest files; finite reference math and ordinary CUDA graph replay |
| Benchmark | Per-op benchmark files; latency and baseline/candidate SpeedUp, separate per shape |
| Active shapes | IHC M=1/3/4/8/64/1024, H=6144, HC=4; multiply M=1/4/8/64/256/1024,N=1024; MLA M=1/2/4/8/16/32/64/1024,H=4,padded H=64,D=512+64 |
| Timeout | Suggested finite functional run 180 seconds; no new timing run in delivery audit |
| Profiler | Historical E23 all16 profile confirms actual execution; no new profiler launched |
| Autotune | Mul uses existing scaled_int8_quant_dynamic_single config, 4 warps/1 stage; IHC and concat preserve qualified fixed schedules |
| Fallback | Unsupported metadata raises; production performs no Torch compute fallback |

## Inference API contracts

| API | Input/output and semantics | Unsupported scope |
| --- | --- | --- |
| ihc_post_pre | Canonical CUDA BF16 x[M,6144], residual[M,4,6144], FP32 post[M,4], weight[8,24576], scale[2], base[8]; M=0..1024. Returns BF16 reduced[M,6144], FP32 next gates[M,4], BF16 rounded residual[M,4,6144]. Post scatter is rounded to BF16 before the following RMS/projection reduction. | Other shapes, layouts, devices or training; finite scalar parameters required. No backward API. |
| int8_scaled_mm_decode | SM90 CUDA INT8 A[M,1024], column-major B[1024,6144], FP32 per-row/channel scales; M1..64. Returns BF16[M,6144], preserving scale_b*INT32acc then scale_a then BF16. | Other geometry, architectures, layouts, nonzero offsets, bias or output dtype. No backward. |
| mul_scaled_int8_quant | Equal CUDA BF16 row-major a,b[M,N], N=1..16384, wider row strides allowed. Product rounds to BF16 before FP32 absmax, nearest-even saturating INT8 conversion and FP32 scale[M,1]. Optional output buffers follow the validated disjoint-storage contract. M=0 returns empty outputs. | HIP, non-CUDA, other dtypes, inner strides or training. No broadcasting/backward. |
| concat_pad_mla_q_and_sink | CUDA BF16 nope[M,H,N], rope[M,H or 1,R], FP32 sink[H], caller outputs BF16[M,HP,N+R] and FP32[HP], HP>=H. Copies source bits, fills padded query with +0 and padded sink with -Inf. | Nonpositive/symbolic dimensions, unsafe spans/strides, wrong dtypes, training or invalid outputs. No backward. |

| Path | Trigger | Implementation and config | Checks |
| --- | --- | --- | --- |
| Empty | IHC M=0 or mul M=0 | Metadata plus empty allocation; no kernel | Empty outputs |
| Fast | Documented supported inputs | Two IHC stages; one mul/quant kernel; one concat/pad kernel | Reference math and ordinary graph |
| General | Supported row strides and rope head sharing | Same qualified kernels; metadata routing only | Strided mul and shared-rope cases |
| Unsupported | Outside explicit contract | Raise before launch, no Torch compute fallback | Existing capability/metadata gates |
| External library | None for numerical result | No external compute provider | Static source audit |

INT8 decode preserves the qualified SM90 tile64x64x256,4 warps,3 stages and
scale_b-before-scale_a epilogue, with no new schedule.

Fixed-schedule exemptions preserve the archived reduction tree and rounding.
IHC uses K-splits=24, BLOCK_K=1024, 8 warps then BLOCK_D=1024,4 warps;
concat uses BLOCK=1024,4 warps,1 stage,1 CTA. New schedules need separate
qualification. The helper `_storage_bounds` is copied unchanged from the
archived transport module into concat; the rejected standalone pad operator
and its optional preparation/cache API are not included.

## Source identity and evidence

| API | Archived full-source SHA256 | Existing qualification |
| --- | --- | --- |
| ihc_post_pre | a1cb0cd02e57957206b2b758039c385f86a4918f3ed3a570a1349489c523a0d7 | E8 component batch2 IHC BF16 bits and ordinary graph, M=1/3/8/64/1024; E23 actual profile |
| int8_scaled_mm_decode | 865a04cab00ebf9b94967a7fe962784638938953e785da0d8f164c9bd105a1d3 | E9 output-projection public API math/ordinary graph gate; primitive195 cases and all decode buckets; E23 actual profile |
| mul_scaled_int8_quant | 7745f6307a3c53d460f9055b1c2f01a89407798d4253e5a9389c59079eb2dab8 | E16 gate INT8 finite/nonfinite numerical and ordinary graph admission; E23 actual profile |
| concat_pad_mla_q_and_sink | b7f48759de3a3c9256ea419b7f29c73398aeb553f318255195f79e6a3d9da68b | Final MLA composite r3 admission, 67 numerical/graph cases and all decode buckets; E23 actual profile |

Formatting may change bytes; numerical function ASTs are checked against those
fixed sources. Concat's storage-validation helper is inlined from the same
archived source without a semantic change. The new finite test and benchmark
entry points have not been executed on current main. Historical component
qualification and execution do not establish current-main or full-model
numerical/performance acceptance.

The archived E23 approximately 10.257 ms/token measurement belongs to its entire
framework/plugin/source cohort. These APIs carry no individual serving-speed
attribution. Current distributed numerical acceptance, nonfinite full-model
domains and dataset accuracy remain separate gates. No new benchmark was run
for this packaging change.

## Production Torch audit

Torch calls in the new production paths read metadata or allocate uninitialized
outputs. CUDA device contexts choose the launch device. The mul alias query and
concat allocation-bound checks validate output metadata. Tensor arithmetic,
casts, copies and reductions appear only in test/benchmark references. There is
no matching single aten schema for these fused APIs, so no aten registration is
added. Existing IHC pre/post exports and Ascend attention overrides are preserved.
