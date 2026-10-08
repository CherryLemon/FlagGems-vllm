# SPDX-License-Identifier: Apache-2.0
"""The caller supplies squeezed [N] scales, including N == head_dim."""

import importlib

import pytest
import torch

import flaggems_vllm as gems

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


@pytest.mark.parametrize("tokens", [128, 512, 2051])
def test_prefill_mqa_squeezed_key_scale(tokens):
    keys = torch.ones(tokens, 128, device=gems.device).to(torch.float8_e4m3fn)
    scales = torch.arange(1, tokens + 1, dtype=torch.float32, device=gems.device)
    query = torch.ones(2, 3, 128, device=gems.device).to(torch.float8_e4m3fn)
    weights = torch.ones(2, 3, device=gems.device)
    starts = torch.tensor([0, 1], dtype=torch.int32, device=gems.device)
    ends = torch.tensor([tokens, tokens - 1], dtype=torch.int32, device=gems.device)
    actual = gems.fp8_fp4_mqa_logits(
        (query, None), (keys, scales), weights, starts, ends
    )
    expected = (384 * scales).expand(2, -1).clone()
    expected[1, 0] = expected[1, -1] = -torch.inf
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("invalid_shape", [(3,), (128, 2)])
def test_invalid_key_scale_groups_rejected_before_numerics(invalid_shape, monkeypatch):
    module = importlib.import_module("flaggems_vllm.ops.fp8_fp4_mqa_logits")
    calls = []

    class RejectLaunch:
        def __getitem__(self, grid):
            calls.append(grid)
            raise AssertionError("invalid scale shape launched")

    monkeypatch.setattr(module, "_fp8_fp4_mqa_logits_kernel", RejectLaunch())
    query = torch.ones(2, 3, 128, device=gems.device).to(torch.float8_e4m3fn)
    keys = torch.ones(128, 128, device=gems.device).to(torch.float8_e4m3fn)
    scales = torch.ones(invalid_shape, device=gems.device)
    weights = torch.ones(2, 3, device=gems.device)
    starts = torch.zeros(2, dtype=torch.int32, device=gems.device)
    ends = torch.full((2,), 128, dtype=torch.int32, device=gems.device)
    with pytest.raises(ValueError, match="[Ss]cale"):
        gems.fp8_fp4_mqa_logits((query, None), (keys, scales), weights, starts, ends)
    assert not calls
