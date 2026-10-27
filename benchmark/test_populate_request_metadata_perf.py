# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from flaggems_vllm.ops.request_id_metadata import populate_request_metadata

from . import base


def reference(qsl, output, counts):
    lengths = qsl[1:] - qsl[:-1]
    ids = torch.repeat_interleave(
        torch.arange(counts.numel(), device=qsl.device, dtype=output.dtype),
        lengths,
        output_size=counts.numel(),
    )
    output.copy_(torch.nn.functional.pad(ids, (0, output.numel() - ids.numel())))
    counts.copy_(lengths)


class RequestMetadataBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(b, 64) for b in (1, 4, 64)]

    def get_input_iter(self, dtype):
        for batch, capacity in self.shapes:
            yield (
                torch.arange(batch + 1, dtype=dtype, device=self.device),
                torch.empty(capacity, dtype=dtype, device=self.device),
                torch.empty(batch, dtype=dtype, device=self.device),
            )


@pytest.mark.populate_request_metadata
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required",
)
@torch.inference_mode()
def test_populate_request_metadata_perf():
    bench = RequestMetadataBenchmark(
        op_name="populate_request_metadata",
        torch_op=reference,
        gems_op=populate_request_metadata,
        dtypes=[torch.int32],
    )
    bench.run()
