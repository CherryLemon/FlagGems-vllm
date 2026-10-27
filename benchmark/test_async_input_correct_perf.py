# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from flaggems_vllm.ops.async_input_staging import async_input_correct

from . import base


def reference(packet, computed, accepted, valid, positions, drafts):
    batch = accepted.numel()
    previous = packet[:batch]
    cpu = packet[batch : 2 * batch].int()
    old_drafts = packet[2 * batch : 3 * batch]
    indices = previous.clamp(min=0).long()
    participating = (previous >= 0) & (old_drafts[indices] > 0)
    old = computed[indices]
    count = valid[indices]
    computed.copy_(torch.where(participating, old + count, cpu))
    accepted.copy_(torch.where(participating, count, accepted))
    positions.copy_(previous)
    drafts.copy_(old_drafts.int())


def candidate(packet, computed, accepted, valid, positions, drafts):
    batch = accepted.numel()
    return async_input_correct(
        packet,
        computed,
        accepted,
        valid,
        positions,
        drafts,
        num_maps=0,
        num_reqs=batch,
        prev_capacity=batch,
    )


class AsyncCorrectBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(b,) for b in (1, 4, 64)]

    def get_input_iter(self, dtype):
        for (batch,) in self.shapes:
            previous = torch.arange(batch, dtype=torch.int64, device=self.device)
            packet = torch.cat((previous, previous + 20, torch.ones_like(previous)))
            yield (
                packet,
                torch.arange(batch, dtype=dtype, device=self.device),
                torch.zeros(batch, dtype=dtype, device=self.device),
                torch.ones(batch, dtype=dtype, device=self.device),
                torch.empty(batch, dtype=torch.int64, device=self.device),
                torch.empty(batch, dtype=dtype, device=self.device),
            )


@pytest.mark.async_input_correct
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required",
)
@torch.inference_mode()
def test_async_input_correct_perf():
    bench = AsyncCorrectBenchmark(
        op_name="async_input_correct",
        torch_op=reference,
        gems_op=candidate,
        dtypes=[torch.int32],
    )
    bench.run()
