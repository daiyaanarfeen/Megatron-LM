# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Integration tests for token dispatch with heterogeneous MoE replicas.

Verifies that MoELayer construction and AlltoAll token dispatch work correctly
when different replicas have different ep sizes, as created by
initialize_heterogeneous_model_parallel().

Requires WORLD_SIZE=8 GPUs.
"""

import dataclasses

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


def _setup_heterogeneous_groups(tp, cp, k, etp, num_moe_experts):
    """Initialize heterogeneous model parallel groups."""
    parallel_state.destroy_model_parallel()
    Utils.initialize_distributed()
    parallel_state.initialize_heterogeneous_model_parallel(
        tensor_model_parallel_size=tp,
        context_parallel_size=cp,
        num_tp_cp_per_replica=k,
        expert_tensor_parallel_size=etp,
        num_moe_experts=num_moe_experts,
    )
    _set_random_seed(seed_=123, data_parallel_random_init=False)


def _make_config(tp, etp, num_moe_experts, hidden_size=16):
    """Create a TransformerConfig for MoE testing."""
    # expert_model_parallel_size in config is used only for validation,
    # not for actual expert splitting (that comes from process groups).
    # Set it to the ep of the current rank's group.
    ep_size = torch.distributed.get_world_size(
        parallel_state.get_expert_model_parallel_group()
    )
    return TransformerConfig(
        tensor_model_parallel_size=tp,
        expert_model_parallel_size=ep_size,
        pipeline_model_parallel_size=1,
        expert_tensor_parallel_size=etp,
        moe_router_topk=2,
        num_moe_experts=num_moe_experts,
        moe_router_load_balancing_type="aux_loss",
        moe_token_dispatcher_type="alltoall",
        moe_aux_loss_coeff=0.1,
        num_layers=1,
        moe_router_dtype="fp32",
        hidden_size=hidden_size,
        num_attention_heads=8,
        use_cpu_initialization=True,
        sequence_parallel=tp > 1,
    )


def _make_moe_layer(config, dtype=torch.float32):
    """Create an MoELayer from config using default (global) process groups."""
    submodules = get_gpt_layer_local_submodules(
        num_experts=config.num_moe_experts, moe_grouped_gemm=False,
    )
    layer = MoELayer(config, submodules.mlp.submodules).cuda().to(dtype=dtype)
    layer.set_layer_number(0)
    return layer


class TestHeterogeneousMoELayerConstruction:
    """Step 2: Verify MoELayer construction with heterogeneous ep sizes."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.flaky
    @pytest.mark.flaky_in_dev
    @pytest.mark.timeout(120)
    def test_moe_layer_num_local_experts(self):
        """8 GPUs, tp=2, cp=1, k=[1,3], etp=2, num_experts=6.

        Replica 0 (ranks 0-1): ep=1 → 6 local experts
        Replica 1 (ranks 2-7): ep=3 → 2 local experts
        """
        _setup_heterogeneous_groups(tp=2, cp=1, k=[1, 3], etp=2, num_moe_experts=6)

        config = _make_config(tp=2, etp=2, num_moe_experts=6)
        moe_layer = _make_moe_layer(config)

        r = torch.distributed.get_rank()

        if r < 2:  # replica 0
            assert moe_layer.num_local_experts == 6
            assert moe_layer.local_expert_indices == list(range(6))
        else:  # replica 1
            ep_rank = torch.distributed.get_rank(
                parallel_state.get_expert_model_parallel_group()
            )
            assert moe_layer.num_local_experts == 2
            expected_offset = ep_rank * 2
            assert moe_layer.local_expert_indices == [
                expected_offset,
                expected_offset + 1,
            ]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.flaky
    @pytest.mark.flaky_in_dev
    @pytest.mark.timeout(120)
    def test_moe_layer_uniform_replicas(self):
        """8 GPUs, tp=2, cp=1, k=[2,2], etp=2, num_experts=8.

        Both replicas: ep=2 → 4 local experts each.
        """
        _setup_heterogeneous_groups(tp=2, cp=1, k=[2, 2], etp=2, num_moe_experts=8)

        config = _make_config(tp=2, etp=2, num_moe_experts=8)
        moe_layer = _make_moe_layer(config)

        assert moe_layer.num_local_experts == 4
        ep_rank = torch.distributed.get_rank(
            parallel_state.get_expert_model_parallel_group()
        )
        expected_offset = ep_rank * 4
        assert moe_layer.local_expert_indices == list(
            range(expected_offset, expected_offset + 4)
        )


class TestHeterogeneousTokenDispatch:
    """Step 3: End-to-end token dispatch with heterogeneous ep sizes.

    Verifies that dispatch → multiply by probs → combine restores original data.
    """

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.flaky
    @pytest.mark.flaky_in_dev
    @pytest.mark.timeout(120)
    def test_dispatch_roundtrip_heterogeneous(self):
        """8 GPUs, tp=2, cp=1, k=[1,3], etp=2, num_experts=6.

        Token dispatch → expert compute → combine should restore original data
        when probs are uniform.
        """
        _setup_heterogeneous_groups(tp=2, cp=1, k=[1, 3], etp=2, num_moe_experts=6)

        config = _make_config(tp=2, etp=2, num_moe_experts=6)
        moe_layer = _make_moe_layer(config)
        dispatcher = moe_layer.token_dispatcher

        bs, seql = 32, 8
        hidden_states = torch.randn(
            (bs, seql, config.hidden_size), dtype=torch.float32,
        ).cuda()
        ans = hidden_states.clone()
        hidden_states.requires_grad = True

        # Route tokens
        probs, indices = apply_module(moe_layer.router)(hidden_states)
        # Uniform probs so dispatch+combine = identity
        probs = torch.ones_like(probs) / moe_layer.router.topk

        # Dispatch
        hidden_states_d, probs_d = dispatcher.dispatch_preprocess(
            hidden_states, indices, probs,
        )
        hidden_states_d, probs_d = dispatcher.token_dispatch(hidden_states_d, probs_d)
        permuted, tokens_per_expert, permuted_probs = dispatcher.dispatch_postprocess(
            hidden_states_d, probs_d,
        )

        # Simulate expert computation (multiply by probs)
        permuted = permuted * permuted_probs.unsqueeze(-1)
        permuted = permuted.to(dtype=torch.float32)

        # Combine
        permuted = dispatcher.combine_preprocess(permuted)
        permuted = dispatcher.token_combine(permuted)
        restored = dispatcher.combine_postprocess(permuted)

        # Reduce across TP (etp) gives a factor of etp_size
        scale = config.expert_tensor_parallel_size
        restored = restored / scale

        torch.testing.assert_close(restored, ans)

        # Verify backward pass
        torch.autograd.backward(restored, hidden_states)
        torch.testing.assert_close(hidden_states.grad, ans)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.flaky
    @pytest.mark.flaky_in_dev
    @pytest.mark.timeout(120)
    def test_dispatch_roundtrip_uniform(self):
        """8 GPUs, tp=2, cp=1, k=[2,2], etp=2, num_experts=8.

        Uniform replicas — sanity check that heterogeneous init produces
        correct behavior for the degenerate (uniform) case.
        """
        _setup_heterogeneous_groups(tp=2, cp=1, k=[2, 2], etp=2, num_moe_experts=8)

        config = _make_config(tp=2, etp=2, num_moe_experts=8)
        moe_layer = _make_moe_layer(config)
        dispatcher = moe_layer.token_dispatcher

        bs, seql = 32, 8
        hidden_states = torch.randn(
            (bs, seql, config.hidden_size), dtype=torch.float32,
        ).cuda()
        ans = hidden_states.clone()
        hidden_states.requires_grad = True

        probs, indices = apply_module(moe_layer.router)(hidden_states)
        probs = torch.ones_like(probs) / moe_layer.router.topk

        # Dispatch
        hidden_states_d, probs_d = dispatcher.dispatch_preprocess(
            hidden_states, indices, probs,
        )
        hidden_states_d, probs_d = dispatcher.token_dispatch(hidden_states_d, probs_d)
        permuted, tokens_per_expert, permuted_probs = dispatcher.dispatch_postprocess(
            hidden_states_d, probs_d,
        )

        permuted = permuted * permuted_probs.unsqueeze(-1)
        permuted = permuted.to(dtype=torch.float32)

        permuted = dispatcher.combine_preprocess(permuted)
        permuted = dispatcher.token_combine(permuted)
        restored = dispatcher.combine_postprocess(permuted)

        scale = config.expert_tensor_parallel_size
        restored = restored / scale

        torch.testing.assert_close(restored, ans)
