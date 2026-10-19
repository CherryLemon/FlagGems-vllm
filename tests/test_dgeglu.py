# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

import flaggems_vllm

from . import accuracy_utils as utils

try:
    from transformer_engine.pytorch import cpp_extensions as tex

    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False


@pytest.mark.dgeglu
@pytest.mark.skipif(not TE_AVAILABLE, reason="transformer engine is not available")
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_dgeglu(shape, dtype):
    if len(shape) == 0:
        # dgeglu does not support 0-dim scalar tensors.
        return

    if shape[-1] % 2 != 0:
        shape = list(shape)
        shape[-1] += 1
        shape = tuple(shape)

    input_tensor = torch.randn(shape, dtype=dtype, device=flaggems_vllm.device)

    grad_output_shape = list(shape)
    grad_output_shape[-1] //= 2
    grad_output = torch.randn(
        tuple(grad_output_shape), dtype=dtype, device=flaggems_vllm.device
    )
    ref_out = tex.dgeglu(grad_output, input_tensor, None)
    ref_out = utils.to_reference(ref_out)
    with flaggems_vllm.use_gems():
        res_out = flaggems_vllm.dgeglu(grad_output, input_tensor)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.dgeglu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(3, 254), (2, 3, 512)])
def test_dgeglu_torch_reference(shape, dtype):
    """Keep math-shim compatibility covered without TransformerEngine."""
    x = torch.randn(shape, dtype=dtype, device=flaggems_vllm.device)
    reference = x.float().detach().requires_grad_(True)
    a, b = reference.chunk(2, dim=-1)
    expected = torch.nn.functional.gelu(a, approximate="tanh") * b
    grad = torch.randn_like(expected).to(dtype)
    expected.backward(grad.float())
    actual = flaggems_vllm.dgeglu(grad, x)
    expected = reference.grad
    utils.gems_assert_close(actual, expected.to(dtype), dtype)
