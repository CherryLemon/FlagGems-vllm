# Hy4-preview E23 routing and SiLU quantization

This change adds the two opt-in specializations already used by archived E23.
It updates existing operators rather than adding another functional API.

| Operator | Change | Qualified domain | Default |
| --- | --- | --- | --- |
| grouped_topk | Fuse sigmoid score production into group scoring; collapse the one-group route into one kernel | FP32 scores with contiguous inner dimension, unit-stride bias; one group/selected group, power-of-two k<=32, experts<=1024; target E256,k8 | FLAG_GEMS_GROUPED_TOPK_SCORE_FUSION=0 and FLAG_GEMS_GROUPED_TOPK_SINGLE_GROUP_FUSION=0 |
| fused_moe | Fuse clamped SiLU with the second per-token INT8 quantizer while materializing the rounded BF16/FP16 activation; preserve the validated mixed registrar's native division semantics | W8A8, per-channel quantization, no block/OCP MX/a2 scale, SILU activation with explicit finite nonnegative clamp, width<=8192 and a recognized live quantization signature; target expert width2048 and rows=8*model M | FLAG_GEMS_W8A8_QUANT_FUSION=off; explicit quant_silu required |

The generic routes and core GEMM kernel remain the current upstream source.
The SiLU payload is the exact qualified pre-output-buffer source, excluding the
later optional output-buffer integration. No M256 geometry or large-M guard is
included. Unsupported mixed registrar signatures use the pre-existing public
route; the new quantizer does not add a Torch compute fallback. The inherited
generic implementation's Torch behavior is unchanged. No aten API is added.

| Evidence | Exact source SHA256 | Math/ordinary graph scope |
| --- | --- | --- |
| E14 single-group component | 53ca427a1f7d7749f230360fd8a39fff1e86ba12ab8530a24d29a799345352a3 | 115 math cases and 23 boundaries, route IDs/weight bits exact, supported ordinary graph output exact; M1/2/4/8/16/32/64/1024 and target E256,k8 |
| E8 batch2 SiLU component | f9e1d738a2f8f82ef190a49a43a3c31ae21145913946b3889ebd1eaaf34c96e5 | 73 numerical/ordinary graph cases; BF16 workspace/INT8 bytes and finite FP32 scale bits exact; target width2048, quant2 rows8..8192 |

The SiLU numerical functions and selection guard match the merged E23 source
c9fa769bce2bf1e7f94512279947e158489f4ba6b986a312f3ff18502d918ca1.
Only that merged source's unrelated output-buffer integration is omitted.
The all16 E23 profile confirms actual execution separately; execution counts
do not replace math/graph qualification.

New finite tests compare baseline/fused values and ordinary graph output, with
an independent mathematical reference. Benchmarks include actual routing shapes
and quant2 rows=8*M with width2048. These new current-main entry points are unrun;
no new GPU, profiler or timing was launched for this delivery.

The existing fixed kernel schedules are preserved as an explicit autotune
exemption: this delivery changes no tiles, warps, stages or reduction tree.
Any future schedule needs separate qualification. Training/backward and model
integration remain outside this inference component qualification. Historical
component speedups and the complete E23 10.257 ms/token cohort do not establish
an individual serving gain or current distributed numerical acceptance.
