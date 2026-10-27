# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from flaggems_vllm.ops.async_input_staging import async_input_gather


@pytest.mark.async_input_gather
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required",
)
@pytest.mark.parametrize("batch", [1, 4, 64])
@torch.inference_mode()
def test_async_input_gather(batch):
    maps = batch * 2
    destinations = torch.arange(maps, dtype=torch.int64)
    sources = torch.cat((torch.arange(batch), torch.arange(batch) * 3 + 1)).long()
    packet = torch.cat((destinations, sources)).cuda()
    sampled_cpu = (torch.arange(batch, dtype=torch.int32) + 100).reshape(batch, 1)
    draft_cpu = (torch.arange(batch * 3, dtype=torch.int64) + 200).reshape(batch, 3)
    sampled, drafts = sampled_cpu.cuda(), draft_cpu.cuda()
    output = torch.empty(maps, dtype=torch.int32, device="cuda")
    expected = torch.cat((sampled_cpu[:, 0], draft_cpu[:, 1].int()))

    def call():
        async_input_gather(
            packet, output, sampled, drafts, num_maps=maps, num_sampled=batch
        )

    call()
    assert torch.equal(output.cpu(), expected)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        call()
        with torch.cuda.graph(graph, stream=stream):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(output.cpu(), expected)
