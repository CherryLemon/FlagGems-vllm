# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Timer failure recovery must measure completed work and propagate errors."""

import importlib
from types import SimpleNamespace

import pytest

libentry = importlib.import_module("flaggems_vllm.utils.libentry")


def test_event_timer_result_is_preserved(monkeypatch):
    result = [0.2, 0.1, 0.3]
    monkeypatch.setattr(libentry.triton.testing, "do_bench", lambda *a, **kw: result)
    assert (
        libentry._do_bench_with_timer_fallback(
            lambda: None, warmup=1, rep=1, quantiles=(0.5, 0.2, 0.8)
        )
        is result
    )


@pytest.mark.parametrize("vendor", ["kunlunxin", "nvidia"])
def test_zero_event_timer_uses_synchronized_elapsed_work(monkeypatch, vendor):
    def broken_timer(*args, **kwargs):
        estimate_ms = 0.0
        return 1 / estimate_ms

    monkeypatch.setattr(libentry.triton.testing, "do_bench", broken_timer)
    monkeypatch.setattr(libentry.runtime, "device", SimpleNamespace(vendor_name=vendor))
    now = [0.0]
    calls = []

    def kernel():
        calls.append("kernel")
        now[0] += 0.001

    def synchronize():
        calls.append("sync")
        now[0] += 0.001

    monkeypatch.setattr(libentry.time, "perf_counter", lambda: now[0])
    monkeypatch.setattr(
        libentry, "torch_device_fn", SimpleNamespace(synchronize=synchronize)
    )
    if vendor == "nvidia":
        with pytest.raises(ZeroDivisionError):
            libentry._do_bench_with_timer_fallback(
                kernel, warmup=2, rep=4, quantiles=(0.5, 0.2, 0.8)
            )
        assert not calls
    else:
        result = libentry._do_bench_with_timer_fallback(
            kernel, warmup=2, rep=4, quantiles=(0.5, 0.2, 0.8)
        )
        assert result == pytest.approx([2.0, 2.0, 2.0])
        assert calls == ["sync", "kernel", "sync", "kernel", "sync", "kernel", "sync"]


def test_kernel_failure_is_not_swallowed(monkeypatch):
    monkeypatch.setattr(
        libentry.runtime, "device", SimpleNamespace(vendor_name="kunlunxin")
    )

    def broken_kernel(*args, **kwargs):
        raise RuntimeError("compile failure")

    monkeypatch.setattr(libentry.triton.testing, "do_bench", broken_kernel)
    with pytest.raises(RuntimeError, match="compile failure"):
        libentry._do_bench_with_timer_fallback(
            lambda: None, warmup=1, rep=1, quantiles=(0.5, 0.2, 0.8)
        )


def test_kernel_zero_division_is_not_treated_as_zero_event_time(monkeypatch):
    monkeypatch.setattr(
        libentry.runtime, "device", SimpleNamespace(vendor_name="kunlunxin")
    )
    error = ZeroDivisionError("kernel host failure")
    calls = []

    def kernel():
        calls.append("kernel")
        raise error

    def timer(kernel_call, **kwargs):
        estimate_ms = 0.0
        kernel_call()
        return estimate_ms

    monkeypatch.setattr(libentry.triton.testing, "do_bench", timer)
    monkeypatch.setattr(
        libentry,
        "torch_device_fn",
        SimpleNamespace(synchronize=lambda: calls.append("sync")),
    )
    with pytest.raises(ZeroDivisionError) as caught:
        libentry._do_bench_with_timer_fallback(
            kernel, warmup=1, rep=1, quantiles=(0.5, 0.2, 0.8)
        )
    assert caught.value is error
    assert calls == ["kernel"]


@pytest.mark.parametrize(
    "module_name,kernel_names,bench_name",
    [
        (
            "mhc_post",
            ("mhc_post_kernel_hc_mult_4", "mhc_post_kernel_generic"),
            "_mhc_post_do_bench",
        ),
        (
            "mhc_pre",
            ("mhc_pre_fused_kernel_hc_mult_4", "mhc_pre_generic_kernel"),
            "_mhc_pre_do_bench",
        ),
    ],
)
def test_mhc_autotuners_share_timer_recovery(
    monkeypatch, module_name, kernel_names, bench_name
):
    module = importlib.import_module(f"flaggems_vllm.ops.mhc.{module_name}")
    bench = getattr(module, bench_name)
    for kernel_name in kernel_names:
        kernel = getattr(module, kernel_name)
        if module.runtime.device.vendor_name == "kunlunxin":
            assert kernel.do_bench is bench
        else:
            assert kernel.do_bench is not bench
    result = [0.2, 0.1, 0.3]
    calls = []

    def timer(kernel, **kwargs):
        calls.append((kernel, kwargs))
        return result

    monkeypatch.setattr(module, "_do_bench_with_timer_fallback", timer)
    kernel = lambda: None
    assert bench(kernel, (0.5, 0.2, 0.8)) is result
    assert calls == [(kernel, {"warmup": 25, "rep": 100, "quantiles": (0.5, 0.2, 0.8)})]
