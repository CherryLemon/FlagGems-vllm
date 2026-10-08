# SPDX-License-Identifier: Apache-2.0
"""Sparse QSA single/split equivalence, invalid pages and CUDA graph replay."""

from __future__ import annotations

import math

import pytest
import torch

from flaggems_vllm.ops.qwen4 import qsa_attention as ops

from . import conftest as cfg

pytestmark = [
    pytest.mark.qsa_attention,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


def _load_qsa_ops():
    return ops


def _case(
    device: torch.device,
    *,
    rows: int = 8,
    topk: int = 513,
    head_dim: int = 256,
) -> dict[str, torch.Tensor]:
    page_size = 16
    pages = math.ceil(topk / page_size)
    physical_blocks = rows * pages
    # Padding the final dimension preserves stride-one dim access while making
    # page/head strides non-contiguous, matching allocator-owned cache views.
    q_storage = torch.randn(rows, 3, head_dim + 8, dtype=torch.bfloat16, device=device)
    k_storage = torch.randn(
        physical_blocks,
        page_size,
        1,
        head_dim + 8,
        dtype=torch.bfloat16,
        device=device,
    )
    v_storage = torch.randn_like(k_storage)
    q = q_storage[..., :head_dim]
    k_cache = k_storage[..., :head_dim]
    v_cache = v_storage[..., :head_dim]
    # Reverse each request's page order to exercise the page-table indirection.
    table = (
        torch.arange(physical_blocks, dtype=torch.int32, device=device)
        .reshape(rows, pages)
        .flip(1)
    )
    indices = (
        torch.arange(topk, dtype=torch.int32, device=device).expand(rows, -1).clone()
    )
    # One sentinel and one out-of-table token per row must be ignored by both
    # the split and single kernels.
    indices[:, -1] = -1
    indices[:, -2] = pages * page_size + 5
    token_to_req = torch.arange(rows, dtype=torch.int32, device=device)
    gate = torch.randn_like(q)
    return {
        "q": q,
        "k": k_cache,
        "v": v_cache,
        "indices": indices,
        "table": table,
        "token_to_req": token_to_req,
        "gate": gate,
    }


def _workspace(case: dict[str, torch.Tensor], splits: int):
    q = case["q"]
    return (
        torch.empty(
            q.shape[0],
            q.shape[1],
            splits,
            q.shape[2],
            dtype=torch.float32,
            device=q.device,
        ),
        torch.empty(
            q.shape[0],
            q.shape[1],
            splits,
            dtype=torch.float32,
            device=q.device,
        ),
        torch.empty(
            q.shape[0],
            q.shape[1],
            splits,
            dtype=torch.float32,
            device=q.device,
        ),
    )


def _run(ops, case, out, *, workspace=None):
    return ops.qsa_sparse_paged_attention(
        case["q"],
        case["k"],
        case["v"],
        case["indices"],
        case["table"],
        case["token_to_req"],
        256**-0.5,
        out,
        gate=case["gate"],
        split_workspace=workspace,
    )


def _assert_split_error(candidate: torch.Tensor, baseline: torch.Tensor) -> None:
    """Use explicit BF16 error gates instead of a broad relative tolerance."""

    diff = (candidate.float() - baseline.float()).abs()
    flat = diff.flatten()
    max_abs = float(flat.max().item())
    rmse = float(torch.sqrt(torch.mean(diff.square())).item())
    p99 = float(torch.quantile(flat, 0.99).item())
    print(f"qsa split error: max_abs={max_abs:.8g} rmse={rmse:.8g} p99={p99:.8g}")
    assert math.isfinite(max_abs) and math.isfinite(rmse) and math.isfinite(p99)
    # The merge is FP32 and both paths round the gated result to BF16.  These
    # gates leave room for reduction-order ULPs while catching a real ABI or
    # page-layout mismatch (the measured rows64/topk2051 max is <1e-3).
    assert max_abs <= 1.0e-2
    assert rmse <= 1.0e-3
    assert p99 <= 2.0e-3


@pytest.mark.gpu
def test_qsa_split8_matches_single_with_invalid_pages_and_strides():
    ops = _load_qsa_ops()
    device = torch.device("cuda")
    case = _case(device)
    assert case["q"].shape == (8, 3, 256)
    assert case["k"].stride(-1) == 1
    assert case["k"].stride(1) != case["k"].shape[2] * case["k"].shape[3]

    baseline = torch.empty_like(case["q"])
    _run(ops, case, baseline)
    torch.cuda.synchronize()

    candidate = torch.empty_like(case["q"])
    workspace = _workspace(case, 8)
    _run(ops, case, candidate, workspace=workspace)
    torch.cuda.synchronize()
    assert torch.isfinite(candidate.float()).all()
    _assert_split_error(candidate, baseline)

    # The reduction has no atomics and should be repeatable for the same input.
    repeat = torch.empty_like(case["q"])
    _run(ops, case, repeat, workspace=workspace)
    torch.cuda.synchronize()
    assert torch.equal(candidate, repeat)


@pytest.mark.gpu
def test_qsa_split_rows64_invalid_requests_pages_and_gate_extremes():
    """Exercise the measured bucket plus every invalid-output gate."""

    ops = _load_qsa_ops()
    device = torch.device("cuda")
    case = _case(device, rows=64, topk=2051)
    case["token_to_req"][0] = -1
    case["token_to_req"][1] = 64
    case["table"][2, 0] = -1
    case["table"][3, -1] = case["k"].shape[0] + 7
    case["indices"][4].fill_(-1)  # all-invalid selection
    case["gate"][6].fill_(-80)
    case["gate"][7].fill_(80)

    baseline = torch.empty_like(case["q"])
    _run(ops, case, baseline)
    torch.cuda.synchronize()

    candidate = torch.empty_like(case["q"])
    workspace = _workspace(case, 8)
    _run(ops, case, candidate, workspace=workspace)
    torch.cuda.synchronize()
    assert torch.isfinite(candidate.float()).all()
    _assert_split_error(candidate, baseline)
    for row in (0, 1, 4):
        assert torch.equal(candidate[row], torch.zeros_like(candidate[row]))
    assert float(candidate[6].float().abs().max()) < 1.0e-3

    # +80 rounds sigmoid to one, so the gated output must equal the same
    # kernel's ungated result for that valid row.
    ungated = torch.empty_like(case["q"])
    ops.qsa_sparse_paged_attention(
        case["q"],
        case["k"],
        case["v"],
        case["indices"],
        case["table"],
        case["token_to_req"],
        256**-0.5,
        ungated,
        gate=None,
        split_workspace=workspace,
    )
    torch.cuda.synchronize()
    assert float((candidate[7].float() - ungated[7].float()).abs().max()) <= 1.0e-2


def qsa_sparse_paged_attention_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    *,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Reference for sparse GQA over selected logical token positions."""

    rows, query_heads, head_dim = q.shape
    kv_heads = k_cache.shape[2]
    group_size = query_heads // kv_heads
    scale = head_dim**-0.5 if softmax_scale is None else softmax_scale
    output = torch.zeros_like(q)
    page_size = k_cache.shape[1]
    for row in range(rows):
        request = int(token_to_req[row])
        if not (0 <= request < block_table.shape[0]):
            continue
        positions = logical_indices[row].long()
        for query_head in range(query_heads):
            kv_head = query_head // group_size
            keys = []
            values = []
            for position in positions.tolist():
                if position < 0:
                    continue
                page = position // page_size
                offset = position % page_size
                if not (0 <= page < block_table.shape[1]):
                    continue
                physical = int(block_table[request, page])
                if not (0 <= physical < k_cache.shape[0]):
                    continue
                keys.append(k_cache[physical, offset, kv_head].float())
                values.append(v_cache[physical, offset, kv_head].float())
            if not keys:
                continue
            key_tensor = torch.stack(keys)
            value_tensor = torch.stack(values)
            scores = torch.matmul(key_tensor, q[row, query_head].float()) * scale
            probs = torch.softmax(scores, dim=0)
            output[row, query_head] = torch.matmul(probs, value_tensor).to(output.dtype)
    return output


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [128, 256] if cfg.QUICK_MODE else [64, 128, 256])
def test_sparse_attention_matches_independent_reference(dtype, head_dim):
    case = _case(torch.device("cuda"), rows=2, topk=33, head_dim=head_dim)
    for name in ("q", "k", "v", "gate"):
        case[name] = case[name].to(dtype)
    out = torch.empty_like(case["q"])
    _run(ops, case, out)
    ref = qsa_sparse_paged_attention_reference(
        case["q"],
        case["k"],
        case["v"],
        case["indices"],
        case["table"],
        case["token_to_req"],
    )
    ref = ref * torch.sigmoid(case["gate"])
    torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)


def test_sparse_graph_replay_changes_inputs():
    case = _case(torch.device("cuda"))
    workspace = _workspace(case, 8)
    out = torch.empty_like(case["q"])
    for _ in range(3):
        _run(ops, case, out, workspace=workspace)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run(ops, case, out, workspace=workspace)
    case["q"].copy_(torch.randn_like(case["q"]))
    case["gate"].copy_(torch.randn_like(case["gate"]))
    graph.replay()
    expected = torch.empty_like(case["q"])
    _run(ops, case, expected)
    _assert_split_error(out, expected)
