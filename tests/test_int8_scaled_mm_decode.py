# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Qualified SM90 output-projection geometry, finite math and ordinary graph."""
import pytest
import torch

from flaggems_vllm.ops.int8_scaled_mm_decode import int8_scaled_mm_decode


@pytest.mark.int8_scaled_mm_decode
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows", [1, 4, 64])
@torch.inference_mode()
def test_int8_scaled_mm_decode(rows):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("qualified SM90 kernel contract")
    generator = torch.Generator(device="cpu").manual_seed(2026)
    a_cpu = torch.randint(-16, 17, (rows, 1024), generator=generator, dtype=torch.int8)
    b_storage = torch.randint(
        -16, 17, (6144, 1024), generator=generator, dtype=torch.int8
    )
    scale_a_cpu = torch.rand(rows, 1, generator=generator) / 32
    scale_b_cpu = torch.rand(6144, 1, generator=generator) / 32
    a = a_cpu.cuda()
    b = b_storage.cuda().t()
    scale_a, scale_b = scale_a_cpu.cuda(), scale_b_cpu.cuda()
    actual = int8_scaled_mm_decode(a, b, scale_a, scale_b)
    # FP64 integer products/sums are exact for these bounded finite inputs.
    accumulator = (a_cpu.double() @ b_storage.t().double()).float()
    expected = (scale_a_cpu * (scale_b_cpu.t() * accumulator)).bfloat16()
    assert torch.equal(actual.cpu(), expected)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        int8_scaled_mm_decode(a, b, scale_a, scale_b)
        with torch.cuda.graph(graph, stream=stream):
            captured = int8_scaled_mm_decode(a, b, scale_a, scale_b)
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, actual)
