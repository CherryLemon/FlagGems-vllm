# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from flaggems_vllm.ops.request_id_metadata import populate_request_metadata


@pytest.mark.populate_request_metadata
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required",
)
@pytest.mark.parametrize("lengths", [[1, 1, 1], [0, 2, 3], [0, 0, 0], [1] * 64])
@torch.inference_mode()
def test_populate_request_metadata(lengths):
    counts_cpu = torch.tensor(lengths, dtype=torch.int32)
    qsl_cpu = torch.cat((torch.zeros(1, dtype=torch.int32), counts_cpu.cumsum(0).int()))
    qsl = qsl_cpu.cuda()
    request_ids = torch.empty(64, dtype=torch.int32, device="cuda")
    counts = torch.empty(len(lengths), dtype=torch.int32, device="cuda")
    expected_ids = torch.zeros(64, dtype=torch.int32)
    for i, length in enumerate(lengths):
        expected_ids[int(qsl_cpu[i]) : int(qsl_cpu[i + 1])] = i

    def call():
        populate_request_metadata(qsl, request_ids, counts)

    call()
    assert torch.equal(request_ids.cpu(), expected_ids)
    assert torch.equal(counts.cpu(), counts_cpu)
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
    assert torch.equal(request_ids.cpu(), expected_ids)
    assert torch.equal(counts.cpu(), counts_cpu)
