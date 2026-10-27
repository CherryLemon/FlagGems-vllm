# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from flaggems_vllm.ops.async_input_staging import async_input_gather

from . import base


def reference(packet, output, sampled, draft):
    maps, batch = output.numel(), sampled.shape[0]
    destinations, sources = packet[:maps], packet[maps:]
    output[destinations[:batch]] = sampled[sources[:batch], 0]
    output[destinations[batch:]] = draft.flatten()[sources[batch:]].int()


def candidate(packet, output, sampled, draft):
    return async_input_gather(
        packet,
        output,
        sampled,
        draft,
        num_maps=output.numel(),
        num_sampled=sampled.shape[0],
    )


class AsyncGatherBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(b,) for b in (1, 4, 64)]

    def get_input_iter(self, dtype):
        for (batch,) in self.shapes:
            dst = torch.arange(batch * 2, dtype=torch.int64, device=self.device)
            src = torch.cat((torch.arange(batch), torch.arange(batch) * 3 + 1)).to(
                self.device
            )
            yield (
                torch.cat((dst, src)),
                torch.empty(batch * 2, dtype=dtype, device=self.device),
                torch.arange(batch, dtype=dtype, device=self.device).reshape(batch, 1),
                torch.arange(batch * 3, dtype=torch.int64, device=self.device).reshape(
                    batch, 3
                ),
            )


@pytest.mark.async_input_gather
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required",
)
@torch.inference_mode()
def test_async_input_gather_perf():
    bench = AsyncGatherBenchmark(
        op_name="async_input_gather",
        torch_op=reference,
        gems_op=candidate,
        dtypes=[torch.int32],
    )
    bench.run()
