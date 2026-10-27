# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from flaggems_vllm.ops.mtp_tail_state import mtp_tail_step


@pytest.mark.mtp_tail_step
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required",
)
@pytest.mark.parametrize("batch", [4, 64])
@torch.inference_mode()
def test_mtp_tail_step(batch):
    temporal = torch.arange(batch, dtype=torch.int64) + 3
    temporal[0] = 255
    positions = torch.stack((temporal, temporal + 17)).cuda()
    seq_cpu = torch.arange(batch, dtype=torch.int32) + 4
    seq_lens = seq_cpu.cuda()
    table_cpu = torch.arange(batch * 4, dtype=torch.int32).reshape(batch, 4) + 10
    table = table_cpu.cuda()
    qsl = torch.arange(batch + 1, dtype=torch.int32, device="cuda")
    out_p = torch.empty_like(positions)
    out_s = torch.empty_like(seq_lens)
    slots = torch.empty(batch, dtype=torch.int64, device="cuda")
    counts = torch.empty(batch, dtype=torch.int32, device="cuda")
    expected_p = torch.zeros(2, batch, dtype=torch.int64)
    expected_s = torch.zeros(batch, dtype=torch.int32)
    expected_slots = torch.full((batch,), -1, dtype=torch.int64)
    expected_counts = torch.zeros(batch, dtype=torch.int32)
    for row in range(batch - 1):
        pos = int(temporal[row]) + 1
        expected_counts[row] = 1
        if pos >= 256:
            expected_s[row] = 1
        else:
            expected_p[:, row] = pos
            expected_s[row] = min(int(seq_cpu[row]) + 1, 256)
            expected_slots[row] = int(table_cpu[row, pos // 64]) * 64 + pos % 64

    def call():
        mtp_tail_step(
            positions,
            seq_lens,
            table,
            qsl,
            out_p,
            out_s,
            slots,
            counts,
            active_batch=batch - 1,
            block_size=64,
            max_model_len=256,
        )

    call()
    expected = (expected_p, expected_s, expected_slots, expected_counts)
    actual = (out_p, out_s, slots, counts)
    for a, e in zip(actual, expected):
        assert torch.equal(a.cpu(), e)
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
    for a, e in zip(actual, expected):
        assert torch.equal(a.cpu(), e)
