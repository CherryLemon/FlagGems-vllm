# Hy4-preview E23 integer metadata and direct-cache M-RoPE

Five functional contracts are delivered from four existing E23 source modules.
The two async integer operations remain separate contracts. Helpers, framework
copies and graph outputs do not add operator counts. This change adds no new
numerical algorithm and installs no automatic model dispatch or framework hook.
Framework selection, buffer holding and scheduling belong to vllm-plugin-FL.

| Direct API | Contract | Existing exact-source GPU evidence |
| --- | --- | --- |
| async_input_correct | CUDA I64 CPU-validated packet and previous positions; I32 computed/accepted/valid/draft vectors. B<=capacity<=1024. Caller buffers are disjoint; reads old values before in-place correction. | f376dbaa907b720226a5b4058c1ee92820cbfe742c743e70d476979252d0600c; 24 native I64 ABI integer cases, static ordinary graph3 replays and all12 source bindings exact |
| async_input_gather | Same packet, unique valid destination indices; I32 sampled IDs and I32/I64 draft IDs; writes I32 output. No allocation. Packet indices are the caller's validated input contract. | Same f376 source/admission; complete five-integer-output comparison. E23 separately observed the two kernels in eager input preparation |
| mtp_tail_step | Caller-owned CUDA outputs; I64 temporal/spatial position planes and slots, I32 sequence lengths/block table/QSL/counts. Preserve native max-length wrap, inactive-row mask and temporal-plane replication. No KV writes. | 3005941170a4fa4c599bf38e59402a7cddce0c54d5633b6213d2b912065da8fe; ten synthetic native-manager component cases including ordinary graph capture/replay |
| populate_request_metadata | Canonical contiguous CUDA I32 QSL, request-ID and count buffers. Caller guarantees monotonic QSL starting at0, final extent<=ID capacity, and disjoint outputs. Fill every padded token ID with0. No allocation. | c006cb57593104971cc4a191dee3b30fcc04cee35617e4ba28ea0e7012fd89bf; twelve metadata cases, native MLA consumer common graph capture/replay exact |
| mrope_strided_cache | CUDA BF16 gapped-head Q[T,H,64], K[T,1,64], cache[N,64], I64 positions[3,T]; T<=1024,H4/32, adjacent pairs, sections sum32. Native output placement is retained: fresh Q, and native conditional K view behavior. | ac9ea8c9f8bf7e4e8df5098ad99e67360229e3daf2190133a6de738fe5eca0fe; fourteen cases against actual imported native M-RoPE, bits and ordinary graph exact |

All numerical and host function ASTs match the qualified sources. Request
metadata imports Triton directly rather than through vllm.triton_utils; the
kernel and launcher are unchanged, and production no longer gains a vLLM
package import dependency. Other vendors expose an explicit unsupported error
through package exports. These direct APIs are excluded from aten registration.

This delivery preserves the qualified fixed schedules as explicit autotune
exemptions: async correction is one CTA, gather uses128 lanes; tail metadata is
one row/program and1 warp; request metadata uses256 lanes; M-RoPE uses one
token/program and4 warps. New schedules need independent qualification.

Focused finite tests use canonical disjoint buffers and ordinary CUDA graphs.
They add no lease, lifetime, stale-buffer, ownership or race-probe workflow.
Real-shape benchmarks cover request batches1/4/64, the two-plane tail update,
request-ID capacity64, and strided H4 M-RoPE at T4/64/1024. Benchmarks include
reference math/transport; the M-RoPE reference uses the installed native vLLM
implementation. No new timing or GPU run was performed for this packaging.

The async component's static graph does not qualify actual serving capture with
dynamic sampler pointers. The tail component does not qualify a complete
GPUModelRunner constructor, weighted Hy4-preview model, TP16 collectives or
accept/reject execution. Request metadata qualifies the stated native consumer
component, rather than a new full-model run. Current-main test entry points,
distributed numerical acceptance and formal performance remain unrun/open.
E23 execution observations are kept separate from these mathematical proofs.
