# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from math import log2

import pytest
import torch

import megatron.core.parallel_state as ps
from tests.unit_tests.test_utilities import Utils

rank = Utils.rank
world_size = Utils.world_size
test_parallel_order = ['tp-cp-ep-dp-pp', 'tp-cp-pp-ep-dp']


@pytest.mark.parametrize('order', test_parallel_order)
@pytest.mark.flaky
@pytest.mark.flaky_in_dev
def test_initialize_and_destroy_model_parallel(order):
    with pytest.raises(AssertionError):
        assert ps.initialize_model_parallel(order=order)
    Utils.initialize_distributed()
    with pytest.raises(RuntimeError):
        assert ps.initialize_model_parallel(tensor_model_parallel_size=2 * world_size, order=order)
    with pytest.raises(RuntimeError):
        assert ps.initialize_model_parallel(
            pipeline_model_parallel_size=2 * world_size, order=order
        )
    with pytest.raises(RuntimeError):
        assert ps.initialize_model_parallel(
            pipeline_model_parallel_size=world_size,
            tensor_model_parallel_size=world_size,
            order=order,
        )
    with pytest.raises(RuntimeError):
        assert ps.initialize_model_parallel(virtual_pipeline_model_parallel_size=2, order=order)
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=2, pipeline_model_parallel_size=4, order=order
    )

    assert ps.model_parallel_is_initialized()
    assert ps.get_model_parallel_group() is not None
    assert ps.get_tensor_model_parallel_group() is not None
    assert ps.get_pipeline_model_parallel_group() is not None
    assert ps.get_data_parallel_group() is not None
    assert ps.get_expert_model_parallel_group() is not None
    assert ps.get_expert_tensor_parallel_group() is not None
    assert ps.get_expert_data_parallel_group() is not None
    assert ps.get_expert_tensor_model_pipeline_parallel_group() is not None
    Utils.destroy_model_parallel()
    assert ps._MODEL_PARALLEL_GROUP is None


@pytest.mark.parametrize('order', test_parallel_order)
def test_pipeline_parallel_initializations(order):
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=2, pipeline_model_parallel_size=4, order=order
    )
    assert ps.get_pipeline_model_parallel_first_rank() == rank % 2
    assert ps.get_data_parallel_src_rank() == rank
    assert ps.get_pipeline_model_parallel_next_rank() == ((rank + 2) % world_size)
    assert ps.get_pipeline_model_parallel_prev_rank() == ((rank - 2) % world_size)
    Utils.destroy_model_parallel()


@pytest.mark.parametrize('order', test_parallel_order)
def test_data_parallel_initializations(order):
    Utils.initialize_model_parallel(pipeline_model_parallel_size=world_size, order=order)
    assert ps.get_data_parallel_src_rank() == rank
    assert ps.get_data_parallel_world_size() == 1
    assert ps.get_data_parallel_rank() == 0
    Utils.destroy_model_parallel()


@pytest.mark.parametrize('order', test_parallel_order)
def test_tensor_model_parellel_world_size(order):
    Utils.initialize_model_parallel(tensor_model_parallel_size=world_size, order=order)
    assert ps.get_tensor_model_parallel_world_size() == world_size
    ps.set_tensor_model_parallel_world_size(None)
    assert ps.get_tensor_model_parallel_world_size() == world_size
    Utils.destroy_model_parallel()


@pytest.mark.parametrize('order', test_parallel_order)
def test_expert_tensor_parellel_world_size(order):
    Utils.initialize_model_parallel(expert_tensor_parallel_size=world_size, order=order)
    assert ps.get_expert_tensor_parallel_world_size() == world_size
    ps.set_expert_tensor_parallel_world_size(None)
    assert ps.get_expert_tensor_parallel_world_size() == world_size
    Utils.destroy_model_parallel()


@pytest.mark.parametrize('order', test_parallel_order)
def test_pipeline_model_parallel_world_size(order):
    Utils.initialize_model_parallel(pipeline_model_parallel_size=world_size, order=order)
    assert ps.get_pipeline_model_parallel_world_size() == world_size
    ps.set_pipeline_model_parallel_world_size(None)
    assert ps.get_pipeline_model_parallel_world_size() == world_size
    Utils.destroy_model_parallel()


@pytest.mark.parametrize('order', test_parallel_order)
def test_tensor_model_parallel_rank(order):
    Utils.initialize_model_parallel(tensor_model_parallel_size=world_size, order=order)
    assert ps.get_tensor_model_parallel_rank() == rank
    ps.set_tensor_model_parallel_rank(None)
    assert ps.get_tensor_model_parallel_rank() == rank
    Utils.destroy_model_parallel()


@pytest.mark.parametrize('order', test_parallel_order)
def test_moe_tensor_model_parellel_rank(order):
    Utils.initialize_model_parallel(expert_tensor_parallel_size=world_size, order=order)
    assert ps.get_expert_tensor_parallel_rank() == rank
    ps.set_expert_tensor_parallel_rank(None)
    assert ps.get_expert_tensor_parallel_rank() == rank
    Utils.destroy_model_parallel()


@pytest.mark.parametrize('order', test_parallel_order)
def test_pipeline_model_parallel_rank(order):
    Utils.initialize_model_parallel(pipeline_model_parallel_size=world_size, order=order)
    assert ps.get_pipeline_model_parallel_rank() == rank
    ps.set_pipeline_model_parallel_rank(None)
    assert ps.get_pipeline_model_parallel_rank() == rank
    Utils.destroy_model_parallel()


def test_context_parallel_rank():
    Utils.initialize_model_parallel(context_parallel_size=world_size)
    assert ps.get_context_parallel_rank() == rank
    Utils.destroy_model_parallel()


def test_expert_model_parallel_rank():
    Utils.initialize_model_parallel(expert_model_parallel_size=world_size)
    assert ps.get_expert_model_parallel_rank() == rank
    ps.set_expert_model_parallel_rank(None)
    assert ps.get_expert_model_parallel_rank() == rank
    Utils.destroy_model_parallel()


@pytest.mark.parametrize('order', test_parallel_order)
def test_is_pipeline_first_stage(order):
    Utils.initialize_model_parallel(pipeline_model_parallel_size=world_size, order=order)
    assert ps.is_pipeline_first_stage(ignore_virtual=False) == (rank == 0)
    assert ps.is_pipeline_first_stage() == (rank == 0)
    Utils.destroy_model_parallel()


@pytest.mark.parametrize('order', test_parallel_order)
def test_is_pipeline_last_stage(order):
    Utils.initialize_model_parallel(pipeline_model_parallel_size=world_size, order=order)
    assert ps.is_pipeline_last_stage(ignore_virtual=False) == (rank == world_size - 1)
    assert ps.is_pipeline_last_stage() == (rank == world_size - 1)
    Utils.destroy_model_parallel()


@pytest.mark.parametrize('order', test_parallel_order)
def test_virtual_pipeline_model_parallel_rank(order):
    Utils.initialize_model_parallel(pipeline_model_parallel_size=world_size, order=order)
    ps.set_virtual_pipeline_model_parallel_rank(rank)
    assert ps.get_virtual_pipeline_model_parallel_rank() == rank
    Utils.destroy_model_parallel()


@pytest.mark.parametrize('order', test_parallel_order)
def test_get_tensor_model_parallel_src_rank(order):
    Utils.initialize_model_parallel(tensor_model_parallel_size=world_size, order=order)
    assert ps.get_tensor_model_parallel_src_rank() == ((rank // world_size) * world_size)
    Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.parametrize(
    'src_tp_pp, ep_size',
    [
        ((1, 8), 1),
        ((2, 4), 1),
        ((4, 2), 1),
        ((8, 1), 1),
        ((4, 1), 2),
        ((1, 1), 8),
        ((1, 1), 2),
        ((2, 1), 4),
    ],
)
def test_different_initialize_order_consistency(src_tp_pp, ep_size):
    Utils.initialize_model_parallel(
        *src_tp_pp, expert_model_parallel_size=ep_size, order='tp-ep-dp-pp'
    )
    tp_rank = ps.get_tensor_model_parallel_rank()
    dp_rank = ps.get_data_parallel_rank()
    pp_rank = ps.get_pipeline_model_parallel_rank()
    ep_rank = ps.get_expert_model_parallel_rank()

    tp_g = torch.distributed.get_process_group_ranks(ps.get_tensor_model_parallel_group())
    dp_g = torch.distributed.get_process_group_ranks(ps.get_data_parallel_group(False))
    pp_g = torch.distributed.get_process_group_ranks(ps.get_pipeline_model_parallel_group())
    dp_no_ep_g = torch.distributed.get_process_group_ranks(ps.get_expert_data_parallel_group())
    cp_g = torch.distributed.get_process_group_ranks(ps.get_context_parallel_group())
    mp_g = torch.distributed.get_process_group_ranks(ps.get_model_parallel_group())
    tp_ep_g = torch.distributed.get_process_group_ranks(
        ps.get_expert_tensor_and_model_parallel_group()
    )
    tp_dp_g = torch.distributed.get_process_group_ranks(
        ps.get_tensor_and_data_parallel_group(False)
    )

    Utils.destroy_model_parallel()

    Utils.initialize_model_parallel(
        *src_tp_pp, expert_model_parallel_size=ep_size, order='tp-pp-ep-dp'
    )
    assert tp_rank == ps.get_tensor_model_parallel_rank()
    assert dp_rank == ps.get_data_parallel_rank()
    assert pp_rank == ps.get_pipeline_model_parallel_rank()
    assert ep_rank == ps.get_expert_model_parallel_rank()

    assert tp_g == torch.distributed.get_process_group_ranks(ps.get_tensor_model_parallel_group())
    assert dp_g == torch.distributed.get_process_group_ranks(ps.get_data_parallel_group(False))
    assert pp_g == torch.distributed.get_process_group_ranks(ps.get_pipeline_model_parallel_group())
    assert dp_no_ep_g == torch.distributed.get_process_group_ranks(
        ps.get_expert_data_parallel_group()
    )
    assert cp_g == torch.distributed.get_process_group_ranks(ps.get_context_parallel_group())
    assert mp_g == torch.distributed.get_process_group_ranks(ps.get_model_parallel_group())
    assert tp_ep_g == torch.distributed.get_process_group_ranks(
        ps.get_expert_tensor_and_model_parallel_group()
    )
    assert tp_dp_g == torch.distributed.get_process_group_ranks(
        ps.get_tensor_and_data_parallel_group(False)
    )

    Utils.destroy_model_parallel()


@pytest.mark.parametrize(
    'src_tp_pp, ep_size',
    [((1, 2), 1), ((1, 4), 1), ((2, 2), 1), ((1, 2), 2), ((1, 4), 2), ((2, 2), 2)],
)
@pytest.mark.flaky
@pytest.mark.flaky_in_dev
def test_different_initialize_order_unconsistency(src_tp_pp, ep_size):
    Utils.initialize_model_parallel(
        *src_tp_pp, expert_model_parallel_size=ep_size, order='tp-ep-dp-pp'
    )

    tp_g = torch.distributed.get_process_group_ranks(ps.get_tensor_model_parallel_group())
    dp_g = torch.distributed.get_process_group_ranks(ps.get_data_parallel_group(False))
    pp_g = torch.distributed.get_process_group_ranks(ps.get_pipeline_model_parallel_group())
    cp_g = torch.distributed.get_process_group_ranks(ps.get_context_parallel_group())
    amax_g = torch.distributed.get_process_group_ranks(ps.get_amax_reduction_group(False))
    mp_g = torch.distributed.get_process_group_ranks(ps.get_model_parallel_group())

    Utils.destroy_model_parallel()

    Utils.initialize_model_parallel(
        *src_tp_pp, expert_model_parallel_size=ep_size, order='tp-pp-ep-dp'
    )
    assert tp_g == torch.distributed.get_process_group_ranks(ps.get_tensor_model_parallel_group())
    assert dp_g != torch.distributed.get_process_group_ranks(ps.get_data_parallel_group(False))
    assert pp_g != torch.distributed.get_process_group_ranks(ps.get_pipeline_model_parallel_group())
    assert cp_g == torch.distributed.get_process_group_ranks(ps.get_context_parallel_group())
    assert amax_g != torch.distributed.get_process_group_ranks(ps.get_amax_reduction_group(False))
    assert mp_g != torch.distributed.get_process_group_ranks(ps.get_model_parallel_group())

    Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.parametrize(
    'nodes, num_gpu, tp, pp, cp, ep',
    [
        (1, 1, 1, 1, 1, 1),
        (1, 8, 8, 1, 1, 1),
        (1, 8, 2, 2, 1, 1),
        (1, 8, 2, 4, 1, 1),
        (3, 8, 8, 3, 1, 1),
        (4, 8, 2, 4, 1, 1),
        (8, 8, 8, 8, 1, 1),
        (8, 8, 2, 1, 1, 4),
        (8, 8, 2, 2, 2, 4),
        (8, 8, 2, 1, 4, 8),
        (8, 8, 2, 2, 2, 8),
        (16, 8, 4, 8, 1, 1),
        (16, 8, 4, 8, 1, 4),
        (16, 8, 4, 8, 4, 1),
        (16, 8, 8, 8, 1, 1),
        (16, 8, 4, 8, 1, 1),
        (16, 8, 8, 8, 1, 1),
        (32, 8, 4, 8, 1, 1),
        (32, 8, 8, 8, 1, 1),
        (32, 8, 4, 8, 1, 4),
        (32, 8, 8, 8, 4, 1),
        (64, 8, 4, 2, 8, 8),
        (64, 8, 4, 8, 1, 1),
        (64, 8, 8, 8, 1, 1),
        (96, 8, 4, 8, 1, 1),
        (128, 8, 4, 2, 8, 8),
        (128, 8, 4, 8, 1, 1),
        (256, 8, 4, 8, 1, 1),
        (316, 8, 4, 8, 1, 1),
        (384, 8, 4, 8, 1, 1),
        (512, 8, 4, 8, 1, 1),
        (768, 8, 4, 8, 1, 1),
        (1024, 8, 4, 8, 1, 1),
        (1280, 8, 4, 8, 1, 1),
        (1344, 8, 4, 8, 1, 1),
    ],
)
def test_rank_generator_for_tp_dp_pp(nodes, num_gpu, tp, pp, cp, ep):
    def golden_rank_result_from_past_code(
        world_size: int,
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        context_parallel_size: int = 1,
        expert_model_parallel_size: int = 1,
    ):
        data_parallel_size: int = world_size // (
            tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
        )
        num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
        num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size

        dp_groups = []
        dp_groups_with_cp = []

        all_data_parallel_group_ranks_with_cp = []
        for i in range(pipeline_model_parallel_size):
            start_rank = i * num_pipeline_model_parallel_groups
            end_rank = (i + 1) * num_pipeline_model_parallel_groups
            for j in range(context_parallel_size * tensor_model_parallel_size):
                ranks = range(
                    start_rank + j, end_rank, context_parallel_size * tensor_model_parallel_size
                )
                dp_groups.append(list(ranks))
            for j in range(tensor_model_parallel_size):
                ranks_with_cp = range(start_rank + j, end_rank, tensor_model_parallel_size)
                all_data_parallel_group_ranks_with_cp.append(list(ranks_with_cp))
                dp_groups_with_cp.append(list(ranks_with_cp))

        cp_group = []
        for i in range(pipeline_model_parallel_size):
            for j in range(data_parallel_size):
                start_rank = (
                    i * num_pipeline_model_parallel_groups
                    + j * tensor_model_parallel_size * context_parallel_size
                )
                end_rank = (
                    i * num_pipeline_model_parallel_groups
                    + (j + 1) * tensor_model_parallel_size * context_parallel_size
                )
                for k in range(tensor_model_parallel_size):
                    ranks = range(start_rank + k, end_rank, tensor_model_parallel_size)
                    cp_group.append(list(ranks))

        mp_group = []
        for i in range(data_parallel_size * context_parallel_size):
            ranks = [
                data_parallel_group_ranks_with_cp[i]
                for data_parallel_group_ranks_with_cp in all_data_parallel_group_ranks_with_cp
            ]
            mp_group.append(list(ranks))

        tp_group = []
        for i in range(num_tensor_model_parallel_groups):
            ranks = range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
            tp_group.append(list(ranks))

        pp_group = []
        for i in range(num_pipeline_model_parallel_groups):
            ranks = range(i, world_size, num_pipeline_model_parallel_groups)
            pp_group.append(list(ranks))

        tp_dp_group = []
        tp_dp_cp_group = []
        tensor_and_data_group_size_with_cp: int = (
            tensor_model_parallel_size * data_parallel_size * context_parallel_size
        )
        num_tensor_and_data_groups_with_cp: int = world_size // tensor_and_data_group_size_with_cp
        for i in range(num_tensor_and_data_groups_with_cp):
            start_rank = i * tensor_and_data_group_size_with_cp
            end_rank = start_rank + tensor_and_data_group_size_with_cp
            ranks = range(start_rank, end_rank)
            tp_dp_cp_group.append(list(ranks))

            for j in range(context_parallel_size):
                ranks = []
                for k in range(data_parallel_size):
                    start_rank = (
                        i * tensor_and_data_group_size_with_cp
                        + j * tensor_model_parallel_size
                        + k * tensor_model_parallel_size * context_parallel_size
                    )
                    end_rank = start_rank + tensor_model_parallel_size
                    ranks = ranks + list(range(start_rank, end_rank))
                tp_dp_group.append(list(ranks))

        expert_tp_ep_group = []
        expert_dp_group = []

        expert_data_parallel_size = world_size // (
            tensor_model_parallel_size * pipeline_model_parallel_size * expert_model_parallel_size
        )
        all_ranks = torch.arange(world_size).reshape(
            (
                pipeline_model_parallel_size,
                expert_data_parallel_size,
                expert_model_parallel_size,
                tensor_model_parallel_size,
            )
        )
        # (pp, dp, ep, tp) -> (pp*dp, ep*tp)
        tp_ep_rearrange = torch.reshape(
            all_ranks, (-1, expert_model_parallel_size * tensor_model_parallel_size)
        )
        num_tp_ep_groups = tp_ep_rearrange.shape[0]
        for i in range(num_tp_ep_groups):
            expert_tensor_and_model_parallel_ranks = tp_ep_rearrange[i].tolist()
            expert_tp_ep_group.append(expert_tensor_and_model_parallel_ranks)

        # (pp, dp, ep, tp) -> (pp*ep*tp, dp)
        expert_dp_rearrange = torch.permute(all_ranks, (0, 2, 3, 1)).reshape(
            -1, expert_data_parallel_size
        )
        num_expert_dp_groups = world_size // expert_data_parallel_size
        for i in range(num_expert_dp_groups):
            expert_dp_ranks = expert_dp_rearrange[i].tolist()
            expert_dp_group.append(expert_dp_ranks)

        return (
            dp_groups,
            dp_groups_with_cp,
            cp_group,
            mp_group,
            tp_group,
            pp_group,
            tp_dp_group,
            tp_dp_cp_group,
            expert_tp_ep_group,
            expert_dp_group,
        )

    world_size = nodes * num_gpu
    dp = world_size // (tp * pp * cp)
    expert_dp = world_size // (tp * ep * pp)
    assert dp % ep == 0, f"dp size ({dp}) is not divisible by ep {ep} ."
    assert (
        world_size % (tp * pp * cp) == 0
    ), f"world_size ({world_size}) is not divisible by tp {tp} x pp {pp} x cp {cp}."
    (
        dp_groups,
        dp_groups_with_cp,
        cp_group,
        mp_group,
        tp_group,
        pp_group,
        tp_dp_group,
        tp_dp_cp_group,
        expert_tp_ep_group,
        expert_dp_group,
    ) = golden_rank_result_from_past_code(
        world_size=world_size,
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        context_parallel_size=cp,
        expert_model_parallel_size=ep,
    )
    rank_generator = ps.RankGenerator(tp=tp, ep=1, dp=dp, pp=pp, cp=cp, order="tp-cp-dp-pp")
    expert_rank_generator = ps.RankGenerator(
        tp=tp, ep=ep, dp=expert_dp, pp=pp, cp=1, order="tp-ep-dp-pp"
    )
    assert dp_groups == rank_generator.get_ranks(
        "dp"
    ), f"{dp_groups} != {rank_generator.get_ranks('dp')}"
    assert dp_groups_with_cp == rank_generator.get_ranks(
        'dp-cp'
    ), f"{dp_groups_with_cp} != {rank_generator.get_ranks('dp-cp')}"
    assert cp_group == rank_generator.get_ranks(
        "cp"
    ), f"{cp_group} != {rank_generator.get_ranks('cp')}."
    assert mp_group == rank_generator.get_ranks(
        "tp-pp"
    ), f"{mp_group} != {rank_generator.get_ranks('tp-pp')}"
    assert tp_group == rank_generator.get_ranks(
        "tp"
    ), f"{tp_group} != {rank_generator.get_ranks('tp')}"
    assert pp_group == rank_generator.get_ranks(
        "pp"
    ), f"{pp_group} != {rank_generator.get_ranks('pp')}"
    assert tp_dp_group == rank_generator.get_ranks(
        "tp-dp"
    ), f"{tp_dp_group} != {rank_generator.get_ranks('tp-dp')}"
    assert tp_dp_cp_group == rank_generator.get_ranks(
        "tp-dp-cp"
    ), f"{tp_dp_cp_group} != {rank_generator.get_ranks('tp-dp-cp')}"
    assert expert_tp_ep_group == expert_rank_generator.get_ranks(
        "tp-ep"
    ), f"{expert_tp_ep_group} != {expert_rank_generator.get_ranks('tp-ep')}."
    assert expert_dp_group == expert_rank_generator.get_ranks(
        "dp"
    ), f"{expert_dp_group} != {expert_rank_generator.get_ranks('dp')}."


@pytest.mark.parametrize(
    "world_size, tp_size, cp_size, dp_size",
    [(8, 1, 2, 4), (8, 1, 1, 8)],  # 8 GPUs, 1 TP, 2 CP, 4 DP  # 8 GPUs, 1 TP, 1 CP, 8 DP
)
def test_hybrid_dp_cp_groups(world_size, tp_size, cp_size, dp_size):
    """
    Test that hybrid DPxCP groups are created correctly.
    """
    Utils.destroy_model_parallel()

    # Skip if world size doesn't match
    actual_world_size = torch.cuda.device_count()
    if actual_world_size != world_size:
        pytest.skip(f"Test requires world_size={world_size}, but got {actual_world_size}")
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        context_parallel_size=cp_size,
        hybrid_context_parallel=True,
    )

    dp_cp_size = ps.get_data_parallel_world_size(with_context_parallel=True)
    group_sizes = [2**i for i in range(int(log2(dp_cp_size)))][1:]
    for group_size in group_sizes:
        group = ps.get_hybrid_data_context_parallel_groups(group_size=group_size)
        assert group.size() == group_size

    Utils.destroy_model_parallel()


def test_separate_all_gather_group():
    """Test separate all-gather group for improved communication overlap."""
    # Test without creating AG group (default)
    Utils.initialize_model_parallel(context_parallel_size=world_size, create_all_gather_group=False)
    assert not ps.has_separate_all_gather_group()
    assert ps._DATA_PARALLEL_GROUP_WITH_CP_AG is None
    Utils.destroy_model_parallel()

    # Test with creating AG group
    Utils.initialize_model_parallel(context_parallel_size=world_size, create_all_gather_group=True)
    assert ps.has_separate_all_gather_group()
    assert ps._DATA_PARALLEL_GROUP_WITH_CP_AG is not None

    # Verify it returns the correct group
    ag_group = ps.get_data_parallel_group(with_context_parallel=True, independent_all_gather=True)
    regular_group = ps.get_data_parallel_group(
        with_context_parallel=True, independent_all_gather=False
    )
    assert ag_group is not None
    assert regular_group is not None
    # They should have the same ranks but different communicators
    ag_ranks = torch.distributed.get_process_group_ranks(ag_group)
    regular_ranks = torch.distributed.get_process_group_ranks(regular_group)
    assert ag_ranks == regular_ranks

    Utils.destroy_model_parallel()


# ── HeterogeneousRankGenerator tests (pure unit tests, no distributed init) ──


def _all_ranks(world_size):
    """Return set of all ranks."""
    return set(range(world_size))


def _assert_partition(groups, expected_ranks):
    """Assert groups form an exact partition of expected_ranks (no gaps, no dupes)."""
    all_ranks = []
    for g in groups:
        all_ranks.extend(g)
    assert sorted(all_ranks) == sorted(expected_ranks), (
        f"Groups do not partition expected ranks.\n"
        f"  Got: {sorted(all_ranks)}\n"
        f"  Expected: {sorted(expected_ranks)}"
    )


@pytest.mark.internal
def test_heterogeneous_rank_generator_basic():
    """8 GPUs, tp=2, cp=1, k=[1,3], etp=2 — matches plan example."""
    gen = ps.HeterogeneousRankGenerator(tp=2, cp=1, num_tp_cp_per_replica=[1, 3], etp=2)

    assert gen.world_size == 8
    assert gen.dp == 4
    assert gen.min_k == 1
    assert gen.num_replicas == 2
    assert gen.replica_offsets == [0, 2, 8]

    # Attention groups
    assert gen.get_ranks('tp') == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert gen.get_ranks('dp') == [[0, 2, 4, 6], [1, 3, 5, 7]]
    assert gen.get_ranks('dp-cp') == [[0, 2, 4, 6], [1, 3, 5, 7]]  # cp=1
    assert gen.get_ranks('tp-dp') == [[0, 1, 2, 3, 4, 5, 6, 7]]
    assert gen.get_ranks('tp-dp-cp') == [[0, 1, 2, 3, 4, 5, 6, 7]]

    # Expert groups
    assert gen.get_ranks('etp-ep') == [[0, 1], [2, 3, 4, 5, 6, 7]]
    assert gen.get_ranks('etp') == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert gen.get_ranks('ep') == [[0], [1], [2, 4, 6], [3, 5, 7]]
    assert gen.get_ranks('edp') == [[0, 2], [1, 3]]


@pytest.mark.internal
def test_heterogeneous_rank_generator_with_cp():
    """16 GPUs, tp=2, cp=2, k=[1,3], etp=2."""
    gen = ps.HeterogeneousRankGenerator(tp=2, cp=2, num_tp_cp_per_replica=[1, 3], etp=2)

    assert gen.world_size == 16
    assert gen.tp_cp == 4
    assert gen.dp == 4  # total tp*cp units
    assert gen.min_k == 1
    assert gen.replica_offsets == [0, 4, 16]

    # Replica 0: ranks 0-3 (1 tp*cp unit)
    #   tp*cp unit: [0,1,2,3] where rank = tp_rank + cp_rank*2
    #   tp groups: [0,1], [2,3]
    #   cp groups: [0,2], [1,3]
    # Replica 1: ranks 4-15 (3 tp*cp units)
    #   Unit 0: [4,5,6,7], Unit 1: [8,9,10,11], Unit 2: [12,13,14,15]

    # tp: 8 groups of size 2
    tp_groups = gen.get_ranks('tp')
    assert len(tp_groups) == 8
    assert tp_groups[0] == [0, 1]
    assert tp_groups[1] == [2, 3]
    assert tp_groups[2] == [4, 5]

    # cp: 8 groups of size 2
    cp_groups = gen.get_ranks('cp')
    assert len(cp_groups) == 8
    assert cp_groups[0] == [0, 2]  # tp_rank=0 within unit 0
    assert cp_groups[1] == [1, 3]  # tp_rank=1 within unit 0

    # dp (without cp): ranks with same tp_rank and cp_rank across all units
    dp_groups = gen.get_ranks('dp')
    assert len(dp_groups) == 4  # tp*cp = 4 groups
    # dp=4, each group size 4
    for g in dp_groups:
        assert len(g) == 4

    # dp-cp: ranks with same tp_rank across all units (dp*cp per group)
    dp_cp_groups = gen.get_ranks('dp-cp')
    assert len(dp_cp_groups) == 2  # tp=2 groups
    for g in dp_cp_groups:
        assert len(g) == 8  # dp*cp = 4*2 = 8

    # Expert groups
    etp_ep = gen.get_ranks('etp-ep')
    assert etp_ep == [[0, 1, 2, 3], [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]

    etp_groups = gen.get_ranks('etp')
    assert len(etp_groups) == 8  # 1*4/2 + 3*4/2 = 2 + 6 = 8
    assert etp_groups[0] == [0, 1]
    assert etp_groups[1] == [2, 3]

    ep_groups = gen.get_ranks('ep')
    # Replica 0: ep=2, etp=2 → 2 ep groups of size 2
    assert ep_groups[0] == [0, 2]
    assert ep_groups[1] == [1, 3]
    # Replica 1: ep=6, etp=2 → 2 ep groups of size 6
    assert ep_groups[2] == [4, 6, 8, 10, 12, 14]
    assert ep_groups[3] == [5, 7, 9, 11, 13, 15]

    # edp: min_k=1, min_k*tp_cp=4 groups, each of size 2
    edp_groups = gen.get_ranks('edp')
    assert len(edp_groups) == 4
    assert edp_groups == [[0, 4], [1, 5], [2, 6], [3, 7]]


@pytest.mark.internal
def test_heterogeneous_rank_generator_etp_less_than_tp():
    """8 GPUs, tp=4, cp=1, etp=2, k=[1,1]."""
    gen = ps.HeterogeneousRankGenerator(tp=4, cp=1, num_tp_cp_per_replica=[1, 1], etp=2)

    assert gen.world_size == 8
    assert gen.tp_cp == 4
    assert gen.dp == 2

    # tp groups (attention): size 4
    assert gen.get_ranks('tp') == [[0, 1, 2, 3], [4, 5, 6, 7]]

    # etp groups: size 2 (smaller than tp)
    assert gen.get_ranks('etp') == [[0, 1], [2, 3], [4, 5], [6, 7]]

    # ep groups: etp=2 columns, ep=4/2=2 per replica
    assert gen.get_ranks('ep') == [[0, 2], [1, 3], [4, 6], [5, 7]]

    # etp-ep: full replicas
    assert gen.get_ranks('etp-ep') == [[0, 1, 2, 3], [4, 5, 6, 7]]

    # edp: min_k=1, tp_cp=4 → 4 groups of size 2
    assert gen.get_ranks('edp') == [[0, 4], [1, 5], [2, 6], [3, 7]]


@pytest.mark.internal
def test_heterogeneous_rank_generator_uniform_degenerate():
    """k=[2,2,2] with tp=2 should match homogeneous RankGenerator."""
    gen = ps.HeterogeneousRankGenerator(tp=2, cp=1, num_tp_cp_per_replica=[2, 2, 2])

    assert gen.world_size == 12
    assert gen.dp == 6

    # Compare with homogeneous expert generator: etp=2, ep=2, edp=3
    homogeneous = ps.RankGenerator(tp=2, ep=2, dp=3, pp=1, cp=1, order="tp-ep-dp")

    # etp-ep should match tp-ep
    assert sorted(map(tuple, gen.get_ranks('etp-ep'))) == sorted(
        map(tuple, homogeneous.get_ranks('tp-ep'))
    )
    # etp should match tp
    assert sorted(map(tuple, gen.get_ranks('etp'))) == sorted(
        map(tuple, homogeneous.get_ranks('tp'))
    )
    # ep should match ep
    assert sorted(map(tuple, gen.get_ranks('ep'))) == sorted(
        map(tuple, homogeneous.get_ranks('ep'))
    )
    # edp should match dp
    assert sorted(map(tuple, gen.get_ranks('edp'))) == sorted(
        map(tuple, homogeneous.get_ranks('dp'))
    )


@pytest.mark.internal
@pytest.mark.parametrize(
    'tp, cp, etp, k',
    [
        (2, 1, 2, [1, 3]),
        (2, 1, 2, [2, 2]),
        (2, 2, 2, [1, 3]),
        (4, 1, 2, [1, 1]),
        (4, 1, 4, [1, 1, 2]),
        (2, 1, 2, [1, 1, 1, 1]),
        (4, 2, 2, [1, 3]),
        (2, 1, 1, [2, 2]),
        (8, 1, 4, [1, 2, 3]),
    ],
)
def test_heterogeneous_rank_generator_partition_invariant(tp, cp, etp, k):
    """Verify groups partition all ranks correctly for all keys."""
    gen = ps.HeterogeneousRankGenerator(tp=tp, cp=cp, num_tp_cp_per_replica=k, etp=etp)
    all_expected = list(range(gen.world_size))

    # Attention keys: must partition all ranks
    for key in ['tp', 'cp', 'dp', 'dp-cp', 'tp-dp', 'tp-dp-cp', 'tp-cp']:
        _assert_partition(gen.get_ranks(key), all_expected)

    # Expert keys: etp, ep, etp-ep must partition all ranks
    for key in ['etp', 'ep', 'etp-ep']:
        _assert_partition(gen.get_ranks(key), all_expected)

    # edp: covers exactly min_k * tp_cp * num_replicas ranks
    edp_groups = gen.get_ranks('edp')
    assert len(edp_groups) == gen.min_k * gen.tp_cp
    for g in edp_groups:
        assert len(g) == gen.num_replicas
    edp_ranks = [r for g in edp_groups for r in g]
    assert len(edp_ranks) == len(set(edp_ranks)), "edp groups contain duplicate ranks"
    # Verify edp ranks are from the first min_k tp_cp units of each replica
    for r_idx in range(gen.num_replicas):
        replica_start = gen.replica_offsets[r_idx]
        expected_edp_ranks = set(range(replica_start, replica_start + gen.min_k * gen.tp_cp))
        actual_edp_ranks = {
            rank
            for g in edp_groups
            for rank in g
            if gen.replica_offsets[r_idx] <= rank < gen.replica_offsets[r_idx + 1]
        }
        assert actual_edp_ranks == expected_edp_ranks, (
            f"Replica {r_idx}: expected edp ranks {expected_edp_ranks}, got {actual_edp_ranks}"
        )


@pytest.mark.internal
def test_heterogeneous_rank_generator_with_rank_offset():
    """Verify rank_offset shifts all ranks correctly."""
    offset = 100
    gen = ps.HeterogeneousRankGenerator(
        tp=2, cp=1, num_tp_cp_per_replica=[1, 3], etp=2, rank_offset=offset
    )

    for key in ['tp', 'dp', 'etp', 'ep', 'etp-ep', 'edp']:
        for group in gen.get_ranks(key):
            for rank in group:
                assert rank >= offset, f"Rank {rank} < offset {offset} for key '{key}'"


@pytest.mark.internal
def test_heterogeneous_rank_generator_single_replica():
    """Single replica k=[4] — edp groups each have size 1."""
    gen = ps.HeterogeneousRankGenerator(tp=2, cp=1, num_tp_cp_per_replica=[4], etp=2)

    assert gen.world_size == 8
    assert gen.num_replicas == 1

    # etp-ep: single group of all ranks
    assert gen.get_ranks('etp-ep') == [[0, 1, 2, 3, 4, 5, 6, 7]]

    # edp: min_k=4, 4*2=8 groups, each of size 1 (only 1 replica)
    edp_groups = gen.get_ranks('edp')
    assert len(edp_groups) == 8
    for g in edp_groups:
        assert len(g) == 1
