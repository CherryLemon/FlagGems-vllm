# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU adapter tests; GPU checks are opt-in during development."""

import ast
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def adapter():
    source = (
        Path(__file__).parents[1]
        / "src/flaggems_vllm/ops/minimax_m3_sparse_attention.py"
    )
    spec = importlib.util.spec_from_file_location("vllm_sparse_adapter_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def dispatch(monkeypatch):
    calls = []

    def record(*args, **kwargs):
        calls.append((args, kwargs))
        return args[-1]

    monkeypatch.setitem(
        sys.modules,
        "flag_attn",
        SimpleNamespace(
            minimax_m3_sparse_attn_paged=record,
            minimax_m3_sparse_attn_decode_paged=record,
            minimax_m3_sparse_decode_workspace=record,
        ),
    )
    return calls


@pytest.mark.parametrize("qlen", [1, 2, 4])
def test_decode_uses_no_copy_cache_views(adapter, dispatch, qlen):
    q = torch.empty(2 * qlen, 8, 128)
    cache = torch.empty(4, 2, 128, 1, 128)
    out = torch.empty_like(q)
    scratch = (object(), object())
    result = adapter.minimax_m3_sparse_attn_decode(
        q,
        cache,
        object(),
        object(),
        object(),
        1,
        0.1,
        out,
        qlen,
        workspace=scratch,
        k_scale=0.5,
    )
    args, kwargs = dispatch[0]
    assert result is None
    assert args[1].untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    assert args[2].untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    assert args[1].data_ptr() == cache.select(1, 0).data_ptr()
    assert args[2].data_ptr() == cache.select(1, 1).data_ptr()
    assert args[7] is out and args[8] == qlen
    assert kwargs["workspace"] is scratch and kwargs["k_scale"] == 0.5
    assert kwargs["page_size"] == 128


def test_prefill_forwards_original_metadata(adapter, dispatch):
    q = torch.empty(4, 8, 128)
    cache = torch.empty(4, 2, 128, 1, 128)
    topk, pages, cu, seq, prefix = [object() for _ in range(5)]
    out = torch.empty_like(q)
    assert (
        adapter.minimax_m3_sparse_attn(
            q, cache, topk, pages, cu, seq, prefix, 2, 1, 0.1, out
        )
        is None
    )
    args, _ = dispatch[0]
    assert args[3:8] == (topk, pages, cu, seq, prefix)
    assert args[8:10] == (2, 0.1) and args[10] is out


@pytest.mark.parametrize(
    "shape", [(4, 128, 1, 256), (4, 2, 64, 1, 128), (4, 2, 128, 2, 128)]
)
def test_incompatible_cache_rejected(adapter, dispatch, shape):
    with pytest.raises(ValueError, match="Expected KV"):
        adapter.minimax_m3_sparse_attn_decode(
            torch.empty(2, 8, 128),
            torch.empty(shape),
            None,
            None,
            None,
            1,
            0.1,
            torch.empty(2, 8, 128),
            1,
        )
    assert not dispatch


def test_output_required(adapter, dispatch):
    with pytest.raises(ValueError, match="output buffer"):
        adapter.minimax_m3_sparse_attn_decode(
            torch.empty(2, 8, 128),
            torch.empty(4, 2, 128, 1, 128),
            None,
            None,
            None,
            1,
            0.1,
            None,
            1,
        )
    assert not dispatch


def test_exports_participate_in_existing_registry(adapter):
    root = Path(__file__).parents[1] / "src/flaggems_vllm"
    ops_tree = ast.parse((root / "ops/__init__.py").read_text())
    exports = next(
        ast.literal_eval(n.value)
        for n in ops_tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__all__" for t in n.targets)
    )
    assert set(adapter.__all__) <= set(exports)
    tree = ast.parse((root / "__init__.py").read_text())
    config = next(
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_FULL_CONFIG" for t in n.targets)
    )
    scope = {name: getattr(adapter, name) for name in adapter.__all__}
    scope["_ops_module"] = SimpleNamespace(__all__=exports)
    exec(
        compile(ast.Module(body=[config], type_ignores=[]), "registry_subset", "exec"),
        scope,
    )
    assert {name for name, fn in scope["_FULL_CONFIG"]} == set(adapter.__all__)


@pytest.mark.skipif(
    os.getenv("FLAGGEMS_RUN_SPARSE_GPU_TESTS") != "1",
    reason="GPU regression explicitly deferred",
)
@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gpu_against_torch_reference(decode, dtype):
    import flaggems_vllm
    from flag_attn.testing.minimax_sparse_paged import (
        make_sparse_paged_inputs,
        sparse_paged_reference,
    )

    a = make_sparse_paged_inputs(2, 3, 257, 128, dtype)
    reference = sparse_paged_reference(**{**a, "output": None})
    k, v = a["k_cache"], a["v_cache"]
    # Test construction may copy; production adapter must only select views.
    cache = torch.stack((k, v), dim=1)
    if decode:
        flaggems_vllm.minimax_m3_sparse_attn_decode(
            a["q"],
            cache,
            a["topk_idx"],
            a["block_table"],
            a["seq_lens"],
            1,
            a["sm_scale"],
            a["output"],
            3,
        )
    else:
        flaggems_vllm.minimax_m3_sparse_attn(
            a["q"],
            cache,
            a["topk_idx"],
            a["block_table"],
            a["cu_seqlens_q"],
            a["seq_lens"],
            a["prefix_lens"],
            3,
            1,
            a["sm_scale"],
            a["output"],
        )
    torch.testing.assert_close(a["output"], reference, rtol=0.02, atol=0.02)
