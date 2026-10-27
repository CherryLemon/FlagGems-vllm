# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from flaggems_vllm.ops.mtp_tail_state import mtp_tail_step

from . import base


def reference(p, seq, table, qsl, out_p, out_seq, slots, counts):
    rows = seq.numel()
    active = (torch.arange(rows, device=seq.device) < rows - 1) & (
        qsl[1:] - qsl[:-1] == 1
    )
    position = p[0] + 1
    exceeds = position >= 256
    position = torch.where(exceeds, 0, position)
    block = table.gather(1, (position // 64).clamp(max=3).long()[:, None]).flatten()
    out_p.copy_(torch.where(active, position, 0)[None, :].expand(2, rows))
    out_seq.copy_(
        torch.where(active, torch.where(exceeds, 1, seq + 1).clamp(max=256), 0)
    )
    slots.copy_(torch.where(active & ~exceeds, block * 64 + position % 64, -1))
    counts.copy_(active.int())


def candidate(p, seq, table, qsl, out_p, out_seq, slots, counts):
    return mtp_tail_step(
        p,
        seq,
        table,
        qsl,
        out_p,
        out_seq,
        slots,
        counts,
        active_batch=seq.numel() - 1,
        block_size=64,
        max_model_len=256,
    )


class MTPTailBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(b,) for b in (4, 64)]

    def get_input_iter(self, dtype):
        for (batch,) in self.shapes:
            p = torch.arange(batch, dtype=torch.int64, device=self.device).repeat(2, 1)
            seq = torch.arange(batch, dtype=dtype, device=self.device) + 1
            yield (
                p,
                seq,
                torch.arange(batch * 4, dtype=dtype, device=self.device).reshape(
                    batch, 4
                ),
                torch.arange(batch + 1, dtype=dtype, device=self.device),
                torch.empty_like(p),
                torch.empty_like(seq),
                torch.empty(batch, dtype=torch.int64, device=self.device),
                torch.empty_like(seq),
            )


@pytest.mark.mtp_tail_step
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required",
)
@torch.inference_mode()
def test_mtp_tail_step_perf():
    bench = MTPTailBenchmark(
        op_name="mtp_tail_step",
        torch_op=reference,
        gems_op=candidate,
        dtypes=[torch.int32],
    )
    bench.run()
