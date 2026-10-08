# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors

from types import SimpleNamespace

import pytest

from flaggems_vllm.utils.tle_capabilities import (
    supports_sparse_mla_tle,
    supports_topk_tle,
)


def _language():
    def api(*args, **kwargs):
        raise AssertionError("capability check executed an optional API")

    return SimpleNamespace(
        cumsum=api,
        pipe=api,
        gpu=SimpleNamespace(
            alloc=api, local_ptr=api, copy=api, warp_specialize=api, smem=object()
        ),
    )


@pytest.mark.parametrize(
    "predicate,missing",
    [
        (supports_topk_tle, "cumsum"),
        (supports_topk_tle, "gpu"),
        (supports_topk_tle, "gpu.alloc"),
        (supports_topk_tle, "gpu.local_ptr"),
        (supports_topk_tle, "gpu.smem"),
        (supports_sparse_mla_tle, "pipe"),
        (supports_sparse_mla_tle, "gpu"),
        (supports_sparse_mla_tle, "gpu.alloc"),
        (supports_sparse_mla_tle, "gpu.local_ptr"),
        (supports_sparse_mla_tle, "gpu.copy"),
        (supports_sparse_mla_tle, "gpu.warp_specialize"),
        (supports_sparse_mla_tle, "gpu.smem"),
    ],
)
def test_missing_used_api_rejects_tle_without_mutation(predicate, missing):
    language = _language()
    assert predicate(language)
    owner, _, attr = missing.rpartition(".")
    target = language.gpu if owner else language
    delattr(target, attr)
    before = vars(target).copy()
    assert not predicate(language)
    assert vars(target) == before


@pytest.mark.parametrize("predicate", [supports_topk_tle, supports_sparse_mla_tle])
def test_optional_language_absence(predicate):
    assert not predicate(None)


@pytest.mark.parametrize("attr", ["cumsum", "pipe"])
def test_non_callable_api_rejects_only_its_consumer(attr):
    language = _language()
    setattr(language, attr, object())
    assert supports_topk_tle(language) is (attr != "cumsum")
    assert supports_sparse_mla_tle(language) is (attr != "pipe")
