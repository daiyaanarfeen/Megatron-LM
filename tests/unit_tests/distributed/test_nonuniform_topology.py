# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared topology/rank-ordering tests for opt-in nonuniform TP and EP.

These tests intentionally avoid torch.distributed initialization. They exercise
rank-list helpers only, so NEP and NTP can keep a common global-rank ordering
contract while their process-group setup remains opt-in.
"""

from unittest.mock import Mock, patch

from megatron.core.distributed.nonuniform_common import (
    NonuniformEPRankGenerator,
    NonuniformTPTopologyRankGenerator,
)
from megatron.core.distributed.nonuniform_tp import (
    NonuniformTPConfig,
    _ntp_get_non_active_ranks,
    get_active_ranks_for_dp,
    initialize_nonuniform_tp_process_groups,
)


def _inactive_from_active(tp_base, active_ranks):
    active_rank_set = set(active_ranks)
    return [tp_rank for tp_rank in range(tp_base) if tp_rank not in active_rank_set]


def test_nep_ep32_ep28_edp_groups_align_by_tp_cp_position():
    """NEP uses replica blocks and pairs matching TP*CP positions for EDP."""
    generator = NonuniformEPRankGenerator(
        tp=2,
        cp=2,
        etp=1,
        num_tp_cp_per_replica=[8, 7],
    )

    first_ep_group, second_ep_group = generator.get_ranks('ep')
    edp_groups = generator.get_ranks('edp')

    assert generator.world_size == 60
    assert len(first_ep_group) == 32
    assert len(second_ep_group) == 28
    assert first_ep_group == list(range(0, 32))
    assert second_ep_group == list(range(32, 60))
    assert len(edp_groups) == 28

    for local_position, edp_group in enumerate(edp_groups):
        assert edp_group == [local_position, 32 + local_position]
        assert edp_group[0] % generator.tp_cp == edp_group[1] % generator.tp_cp


def test_nep_rank_generation_is_deterministic_for_shared_topology():
    kwargs = dict(tp=2, cp=2, etp=1, num_tp_cp_per_replica=[8, 7])
    first = NonuniformEPRankGenerator(**kwargs)
    second = NonuniformEPRankGenerator(**kwargs)

    for key in ('ep', 'edp', 'etp', 'etp-ep'):
        assert first.get_ranks(key) == second.get_ranks(key)


def test_ntp_explicit_active_inactive_local_tp_lists_round_trip_by_cp_position():
    """NTP tuple keys preserve the same local TP ordering used by NEP blocks."""
    ntp_config = NonuniformTPConfig(
        tp_base=8,
        tp_spares=2,
        non_active_ranks_per_dp={
            (0, 0, 0): [6, 7],
            (0, 1, 0): [1, 5],
            (1, 0, 0): [0, 3],
            (1, 1, 0): [2, 4],
        },
    )

    for key, expected_non_active in ntp_config.non_active_ranks_per_dp.items():
        dp_rank, cp_rank, pp_rank = key

        assert _ntp_get_non_active_ranks(ntp_config, dp_rank, cp_rank, pp_rank) == (
            expected_non_active
        )
        active_ranks = get_active_ranks_for_dp(
            dp_rank,
            ntp_config.tp_base,
            ntp_config,
            cp_rank=cp_rank,
            pp_rank=pp_rank,
        )
        assert _inactive_from_active(ntp_config.tp_base, active_ranks) == expected_non_active
        assert active_ranks == sorted(active_ranks)


def test_ntp_active_rank_helper_is_deterministic_and_uses_explicit_ordering():
    ntp_config = NonuniformTPConfig(
        tp_base=8,
        tp_spares=3,
        non_active_ranks_per_dp={(0, 0, 0): [1, 4, 6]},
    )

    first = get_active_ranks_for_dp(0, 8, ntp_config, cp_rank=0, pp_rank=0)
    second = get_active_ranks_for_dp(0, 8, ntp_config, cp_rank=0, pp_rank=0)

    assert first == [0, 2, 3, 5, 7]
    assert first == second


def test_ntp_default_inactive_ranks_follow_high_local_tp_rank_convention():
    ntp_config = NonuniformTPConfig(tp_base=8, tp_spares=2)

    active_ranks = get_active_ranks_for_dp(0, 8, ntp_config)

    assert active_ranks == [0, 1, 2, 3, 4, 5]
    assert _inactive_from_active(ntp_config.tp_base, active_ranks) == [6, 7]


@patch('megatron.core.distributed.nonuniform_tp.parallel_state')
@patch('megatron.core.distributed.nonuniform_tp.dist')
def test_ntp_reconfigured_tp_groups_stay_inside_cp_domain_blocks(mock_dist, mock_parallel_state):
    """Current NTP group setup keeps TP groups inside explicit CP-local rank blocks."""
    created_groups = []
    mock_dist.get_rank.return_value = 0
    mock_dist.new_group.side_effect = lambda ranks: created_groups.append(list(ranks)) or Mock()
    mock_parallel_state.get_context_parallel_world_size.return_value = 2
    ntp_config = NonuniformTPConfig(
        tp_base=8,
        tp_spares=2,
        num_reduced_tp_dp_ranks=1,
        non_active_ranks_per_dp={
            (0, 0, 0): [6, 7],
            (0, 1, 0): [6, 7],
        },
    )

    assert initialize_nonuniform_tp_process_groups(ntp_config, exit_spares=False)

    tp_groups = created_groups[:2]
    cp_groups = created_groups[2:8]
    tp_cp_group = created_groups[8]

    assert tp_groups == [list(range(0, 6)), list(range(8, 14))]
    assert all(set(group).issubset(set(range(0, 8))) for group in tp_groups[:1])
    assert all(set(group).issubset(set(range(8, 16))) for group in tp_groups[1:])
    assert cp_groups == [[0, 8], [1, 9], [2, 10], [3, 11], [4, 12], [5, 13]]
    assert tp_cp_group == list(range(0, 6)) + list(range(8, 14))


def test_ntp_topology_rank_generator_keeps_tp_groups_inside_domain_blocks():
    generator = NonuniformTPTopologyRankGenerator(
        tp=8,
        cp=2,
        tp_domain_sizes=[4, 8],
    )

    assert generator.world_size == 24
    assert generator.get_ranks('tp') == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20, 21, 22, 23],
    ]
    assert generator.get_ranks('cp')[:4] == [[0, 4], [1, 5], [2, 6], [3, 7]]
    assert generator.get_ranks('dp')[:4] == [[0, 8], [1, 9], [2, 10], [3, 11]]
    assert generator.get_ranks('dp')[4:8] == [[12], [13], [14], [15]]
    assert generator.get_ranks('tp-cp') == [list(range(0, 8)), list(range(8, 24))]


def test_ntp_topology_rank_metadata_marks_reduced_and_full_domains():
    generator = NonuniformTPTopologyRankGenerator(
        tp=8,
        cp=2,
        tp_domain_sizes=[4, 8],
    )

    assert generator.get_rank_metadata(3) == {
        'replica_index': 0,
        'active_tp_size': 4,
        'tp_rank': 3,
        'cp_rank': 0,
        'is_active': 1,
    }
    assert generator.get_rank_metadata(20) == {
        'replica_index': 1,
        'active_tp_size': 8,
        'tp_rank': 4,
        'cp_rank': 1,
        'is_active': 1,
    }
    assert generator.get_non_active_ranks_per_replica() == {}
