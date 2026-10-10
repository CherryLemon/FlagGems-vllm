# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""vLLM M3 sparse-attention adapters backed by FlagAttention.

KV storage is [pages, 2, 128, kv_heads, head_dim]. Selecting the K/V axis
produces views; cache data is never packed or copied. These are explicit
library APIs, not ATen overloads or automatic patches of vLLM's backend.
"""

__all__ = [
    "minimax_m3_sparse_attn",
    "minimax_m3_sparse_attn_decode",
    "minimax_m3_sparse_decode_workspace",
]


def _cache_views(kv_cache, num_kv_heads, head_dim):
    if kv_cache.ndim != 5 or tuple(kv_cache.shape[1:]) != (
        2,
        128,
        num_kv_heads,
        head_dim,
    ):
        raise ValueError("Expected KV [pages, 2, 128, num_kv_heads, head_dim]")
    return kv_cache.select(1, 0), kv_cache.select(1, 1)


def minimax_m3_sparse_attn(
    q,
    kv_cache,
    topk_idx,
    block_table,
    cu_seqlens_q,
    seq_lens,
    prefix_lens,
    max_query_len,
    num_kv_heads,
    sm_scale,
    output,
    *,
    q_scale=None,
    k_scale=None,
    v_scale=None,
):
    """Match vLLM's prefill signature; write output and return None.

    FP8 backing storage must already be viewed as the intended float8 dtype.
    Metadata values and warmup/capture lifecycle remain the caller's responsibility.
    """
    if output is None:
        raise ValueError("vLLM sparse attention requires an output buffer")
    k, v = _cache_views(kv_cache, num_kv_heads, q.shape[-1])
    from flag_attn import minimax_m3_sparse_attn_paged

    minimax_m3_sparse_attn_paged(
        q,
        k,
        v,
        topk_idx,
        block_table,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        max_query_len,
        sm_scale,
        output,
        page_size=128,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )


def minimax_m3_sparse_attn_decode(
    q,
    kv_cache,
    topk_idx,
    block_table,
    seq_lens,
    num_kv_heads,
    sm_scale,
    output,
    decode_query_len,
    *,
    q_scale=None,
    k_scale=None,
    v_scale=None,
    workspace=None,
):
    """Match vLLM's uniform multi-token decode signature; return None.

    Pass a workspace allocated before capture to make buffer ownership explicit.
    A workspace must be private to concurrent executions/streams.
    """
    if output is None:
        raise ValueError("vLLM sparse attention requires an output buffer")
    k, v = _cache_views(kv_cache, num_kv_heads, q.shape[-1])
    from flag_attn import minimax_m3_sparse_attn_decode_paged

    minimax_m3_sparse_attn_decode_paged(
        q,
        k,
        v,
        topk_idx,
        block_table,
        seq_lens,
        sm_scale,
        output,
        decode_query_len,
        page_size=128,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        workspace=workspace,
    )


def minimax_m3_sparse_decode_workspace(q, num_kv_heads, topk):
    """Allocate shared-library decode scratch before capture."""
    from flag_attn import minimax_m3_sparse_decode_workspace as allocate

    return allocate(q, num_kv_heads, topk)
