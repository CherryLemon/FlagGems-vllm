"""Narrow SM90 W8A8 INT8 decode API; unsupported inputs raise."""

from functools import lru_cache

import torch
import triton
import triton.language as tl


@triton.jit
def scaled_mm_kernel(
    a_ptr,
    b_ptr,
    scale_a_ptr,
    scale_b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    ACCUMULATOR_DTYPE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_SCALE_A: tl.constexpr,
    BLOCK_SIZE_SCALE_B: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    accumulator_dtype = ACCUMULATOR_DTYPE
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=accumulator_dtype)

    # NOTE: Some tensor inputs are so large, they will cause int32 overflow
    # so it is necessary to use tl.int64 for all the offsets, else SEGV will
    # eventually occur.

    # Offsets and masks.
    offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    masks_am = offsets_am < M

    offsets_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    masks_bn = offsets_bn < N

    offsets_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
    offsets_a = stride_am * offsets_am[:, None] + stride_ak * offsets_k[None, :]
    offsets_b = stride_bk * offsets_k[:, None] + stride_bn * offsets_bn[None, :]

    # NOTE: BLOCK_SIZE_SCALE_A could be 1 or BLOCK_SIZE_M, so need to create
    # appropriate offsets and masks for each case. Same goes for
    # BLOCK_SIZE_SCALE_B.
    offsets_scale_am = (
        tl.arange(0, BLOCK_SIZE_SCALE_A)
        + (BLOCK_SIZE_SCALE_A > 1) * pid_m * BLOCK_SIZE_M
    )
    masks_scale_am = offsets_scale_am < M

    offsets_scale_bn = (
        tl.arange(0, BLOCK_SIZE_SCALE_B)
        + (BLOCK_SIZE_SCALE_B > 1) * pid_n * BLOCK_SIZE_N
    )
    masks_scale_bn = offsets_scale_bn < N

    a_ptrs = a_ptr + offsets_a
    b_ptrs = b_ptr + offsets_b

    scale_a_ptrs = scale_a_ptr + offsets_scale_am
    scale_b_ptrs = scale_b_ptr + offsets_scale_bn

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        masks_k = offsets_k < K
        masks_a = masks_am[:, None] & masks_k[None, :]
        a = tl.load(a_ptrs, mask=masks_a)

        masks_b = masks_k[:, None] & masks_bn[None, :]
        b = tl.load(b_ptrs, mask=masks_b)

        # Accumulate results.
        accumulator = tl.dot(a, b, accumulator, out_dtype=accumulator_dtype)

        offsets_k += BLOCK_SIZE_K
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Apply scale at end.
    masks_scale_a = masks_scale_am[:, None] & (tl.arange(0, 1) < 1)[:, None]
    scale_a = tl.load(scale_a_ptrs[:, None], masks_scale_a)
    # Need to broadcast to the appropriate size, if scale_a is already
    # (BLOCK_SIZE_M, 1) then it will broadcast to its own shape. Same goes
    # for scale_b below.
    scale_a = scale_a.broadcast_to((BLOCK_SIZE_M, 1))

    masks_scale_b = masks_scale_bn[:, None] & (tl.arange(0, 1) < 1)[None, :]
    scale_b = tl.load(scale_b_ptrs[:, None], masks_scale_b)
    scale_b = scale_b.broadcast_to((BLOCK_SIZE_N, 1))
    accumulator = scale_b.T * accumulator.to(tl.float32)
    accumulator = scale_a * accumulator

    # Convert to output format.
    c = accumulator.to(c_ptr.type.element_ty)

    # Add bias, it's already in output format, so add it after conversion.
    if bias_ptr:
        offsets_bias = offsets_bn
        bias_ptrs = bias_ptr + offsets_bias
        bias_mask = offsets_bias < N
        bias = tl.load(bias_ptrs, bias_mask)
        c += bias

    # Save output
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    offs_cm = offs_cm.to(tl.int64)
    offs_cn = offs_cn.to(tl.int64)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.store(c_ptrs, c, mask=c_mask)


@lru_cache(maxsize=16)
def _capability(device_index):
    # Hardware metadata only; warmup primes this before FULL capture.
    return torch.cuda.get_device_capability(device_index)


def _require(condition, message, kind=ValueError):
    if not condition:
        raise kind(message)


def int8_scaled_mm_decode(a, b, scale_a, scale_b, out_dtype=torch.bfloat16, bias=None):
    """INT8[M,1024] @ column-major INT8[1024,6144], M1..64 -> BF16.

    Dynamic per-token FP32 scale_a[M,1], channel FP32 scale_b[6144,1],
    no bias. Fresh contiguous output. No streams, owner cache or fallback.
    Native SM90 epilogue order is scale_b*INT32acc -> scale_a -> BF16.
    """
    _require(
        all(isinstance(v, torch.Tensor) for v in (a, b, scale_a, scale_b)),
        "four tensors required",
        TypeError,
    )
    _require(
        a.ndim == b.ndim == scale_a.ndim == scale_b.ndim == 2,
        "2D operands/scales required",
    )
    m, k = a.shape
    _require(
        1 <= m <= 64 and k == 1024 and b.shape == (1024, 6144),
        "unsupported INT8 decode shape",
        NotImplementedError,
    )
    _require(
        a.dtype == b.dtype == torch.int8
        and scale_a.dtype == scale_b.dtype == torch.float32
        and out_dtype == torch.bfloat16
        and bias is None,
        "INT8/FP32/BF16 unbiased contract required",
        TypeError,
    )
    _require(
        all(v.is_cuda and v.device == a.device for v in (a, b, scale_a, scale_b)),
        "one CUDA device required",
        NotImplementedError,
    )
    _require(
        torch.version.hip is None and torch.version.cuda is not None,
        "NVIDIA CUDA component only",
        NotImplementedError,
    )
    _require(
        a.stride() == (1024, 1) and b.stride() == (1, 1024),
        "native A/B strides required",
    )
    _require(
        scale_a.shape == (m, 1)
        and scale_b.shape == (6144, 1)
        and scale_a.stride() == scale_b.stride() == (1, 1),
        "native per-token/channel scale layout required",
    )
    _require(
        all(v.storage_offset() == 0 for v in (a, b, scale_a, scale_b)),
        "zero storage offsets required",
    )
    _require(
        _capability(a.device.index) == (9, 0),
        "SM90 component only",
        NotImplementedError,
    )
    out = torch.empty((m, 6144), dtype=out_dtype, device=a.device)
    scaled_mm_kernel[(96,)](
        a,
        b,
        scale_a,
        scale_b,
        out,
        None,
        m,
        6144,
        1024,
        1024,
        1,
        1,
        1024,
        6144,
        1,
        tl.int32,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=256,
        BLOCK_SIZE_SCALE_A=64,
        BLOCK_SIZE_SCALE_B=64,
        num_warps=4,
        num_stages=3,
        enable_fp_fusion=False,
    )
    return out


__all__ = ["int8_scaled_mm_decode"]
