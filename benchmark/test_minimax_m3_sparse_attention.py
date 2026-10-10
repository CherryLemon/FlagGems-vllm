# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Standard operator benchmark, including decode workspace and merge costs.

The baseline is an unfused Torch reference, not native vLLM performance.
GPU execution is opt-in while this integration is under development.
"""

import os

import flaggems_vllm
import pytest
import torch
from flag_attn.testing.minimax_sparse_paged import (
    make_sparse_paged_inputs,
    sparse_paged_reference,
)

from . import base


def _reference(
    q, cache, topk, pages, cu, seq, prefix, qlen, hk, scale, out, partial, lse
):
    del hk, partial, lse
    return sparse_paged_reference(
        q,
        cache.select(1, 0),
        cache.select(1, 1),
        topk,
        pages,
        cu,
        seq,
        prefix,
        qlen,
        scale,
        out,
    )


class SparseBenchmark(base.Benchmark):
    def __init__(self, decode, **kwargs):
        self.decode = decode
        super().__init__(**kwargs)

    def set_shapes(self, shape_file_path=None):
        self.shapes = (
            [(b, 1, c) for b in (1, 4, 64) for c in (4096, 16384, 65536)]
            if self.decode
            else [(1, q, c) for q in (1, 128, 512) for c in (4096, 16384, 65536)]
        )
        self.shape_desc = "batch, query_len, context_len"

    def set_more_shapes(self):
        return None

    def get_input_iter(self, dtype):
        for batch, qlen, context in self.shapes:
            a = make_sparse_paged_inputs(
                batch, qlen, context, 128, dtype, device=self.device
            )
            cache = torch.stack((a["k_cache"], a["v_cache"]), dim=1)
            partial, lse = flaggems_vllm.minimax_m3_sparse_decode_workspace(
                a["q"], 1, 16
            )
            yield (
                a["q"],
                cache,
                a["topk_idx"],
                a["block_table"],
                a["cu_seqlens_q"],
                a["seq_lens"],
                a["prefix_lens"],
                qlen,
                1,
                a["sm_scale"],
                a["output"],
                partial,
                lse,
            )


@pytest.mark.skipif(
    os.getenv("FLAGGEMS_RUN_SPARSE_GPU_TESTS") != "1", reason="GPU regression deferred"
)
@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.minimax_m3_sparse_attn
def test_minimax_m3_sparse_attention(decode):
    def candidate(
        q, cache, topk, pages, cu, seq, prefix, qlen, hk, scale, out, partial, lse
    ):
        if decode:
            return flaggems_vllm.minimax_m3_sparse_attn_decode(
                q,
                cache,
                topk,
                pages,
                seq,
                hk,
                scale,
                out,
                qlen,
                workspace=(partial, lse),
            )
        return flaggems_vllm.minimax_m3_sparse_attn(
            q, cache, topk, pages, cu, seq, prefix, qlen, hk, scale, out
        )

    bench = SparseBenchmark(
        decode,
        op_name=f"minimax_m3_sparse_{'decode' if decode else 'prefill'}",
        torch_op=_reference,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(candidate)
    bench.run()
