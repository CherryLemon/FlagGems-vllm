# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from flaggems_vllm.ops.async_input_staging import async_input_correct


@pytest.mark.async_input_correct
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required",
)
@pytest.mark.parametrize("batch", [1, 4, 64])
@torch.inference_mode()
def test_async_input_correct(batch):
    previous = torch.arange(batch, dtype=torch.int64)
    previous[-1] = -1
    cpu_counts = torch.arange(batch, dtype=torch.int64) + 20
    drafts = torch.ones(batch, dtype=torch.int64)
    packet = torch.cat((previous, cpu_counts, drafts)).cuda()
    original = torch.arange(batch, dtype=torch.int32) + 4
    valid_cpu = torch.arange(batch, dtype=torch.int32) % 3 + 1
    computed, valid = original.cuda(), valid_cpu.cuda()
    accepted = torch.zeros(batch, dtype=torch.int32, device="cuda")
    positions = torch.empty(batch, dtype=torch.int64, device="cuda")
    out_drafts = torch.empty(batch, dtype=torch.int32, device="cuda")
    expected = original + valid_cpu
    expected[-1] = cpu_counts[-1]
    expected_accepted = valid_cpu.clone()
    expected_accepted[-1] = 0

    def call():
        async_input_correct(
            packet,
            computed,
            accepted,
            valid,
            positions,
            out_drafts,
            num_maps=0,
            num_reqs=batch,
            prev_capacity=batch,
        )

    call()
    assert torch.equal(computed.cpu(), expected)
    assert torch.equal(accepted.cpu(), expected_accepted)
    assert torch.equal(positions.cpu(), previous)
    assert torch.equal(out_drafts.cpu(), drafts.int())
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        call()
        with torch.cuda.graph(graph, stream=stream):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    computed.copy_(original)
    accepted.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(computed.cpu(), expected)
    assert torch.equal(accepted.cpu(), expected_accepted)
