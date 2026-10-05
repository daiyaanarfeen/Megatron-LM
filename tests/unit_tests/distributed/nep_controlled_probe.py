"""Small regression gate; profiling-off runs are explicitly user-authorized."""

import json
import os
import runpy
import sys
from pathlib import Path

NATIVE_ITERATION_MS = []


def install_native_delivery_policy():
    """Disable only healthy expert lookahead; retain native AG and bucket layout."""
    if os.environ['GRAPH_CASE'] != 'healthy':
        return
    no_lookahead = int(os.environ.get('NEP_ABLATION_PREFETCH_DEPTH', '1')) == 0
    from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBucketGroup

    original = _ParamAndGradBucketGroup.finish_param_sync

    def demand_only(self, skip_next_bucket_dispatch=False):
        expert = self.intra_distributed_optimizer_instance_group.group_desc in (
            'ep_dp',
            'EXPERT_DATA_PARALLEL_GROUP',
        )
        return original(
            self, skip_next_bucket_dispatch=skip_next_bucket_dispatch or (expert and no_lookahead)
        )

    _ParamAndGradBucketGroup.finish_param_sync = demand_only


def install_delivery_markers():
    """Measure consumer demand/ready boundaries without extra device synchronization."""
    from cuda_event_capture_adapter import CudaEventCapture

    from megatron.core.distributed.nonuniform_ep import NonuniformEPNCCLParamAndGradBucketGroup
    from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBucketGroup

    class Marker:
        @staticmethod
        def point():
            pass

    CudaEventCapture.wrap(Marker, 'point', 'parameter.boundary')
    healthy = os.environ['GRAPH_CASE'] == 'healthy'
    owner = _ParamAndGradBucketGroup if healthy else NonuniformEPNCCLParamAndGradBucketGroup
    original = owner.finish_param_sync

    def marked(self, *args, **kwargs):
        capture = CudaEventCapture.current
        if capture is None or not capture.active:
            return original(self, *args, **kwargs)
        if healthy:
            if self.param_gather_dispatched and self.param_gather_handle is None:
                return original(self, *args, **kwargs)
            bucket = self.buckets[0].bucket_id
            kind = (
                'expert'
                if self.intra_distributed_optimizer_instance_group.group_desc
                in ('ep_dp', 'EXPERT_DATA_PARALLEL_GROUP')
                else 'dense'
            )
        else:
            if not self._nep_distopt_param_sync_representative:
                return original(self, *args, **kwargs)
            bundle = self._nep_distopt_owner_bundle
            if self.param_gather_dispatched and bundle['param_sync_completed']:
                return original(self, *args, **kwargs)
            bucket, kind = self._nep_nccl_edp_bucket_index, 'expert'

        def point(phase):
            index = len(capture.records)
            Marker.point()
            capture.records[index]['name'] = f'parameter.{phase}.{kind}.bucket{bucket}'

        point('demand')
        result = original(self, *args, **kwargs)
        point('ready')
        return result

    owner.finish_param_sync = marked


def main():
    root = Path(os.environ['GRAPH_OUTPUT'])
    install_native_delivery_policy()
    if os.environ.get('NEP_ACTUAL_A8B') == '1':
        from megatron.core.timers import Timer

        native_elapsed = Timer.elapsed

        def record_elapsed(self, *args, **kwargs):
            value = native_elapsed(self, *args, **kwargs)
            if self.name == 'interval-time' and kwargs.get('barrier'):
                NATIVE_ITERATION_MS.append(value * 1000)
            return value

        Timer.elapsed = record_elapsed
    if 'NEP_CONTROLLED_ROUTING_SEED' in os.environ:
        from nep_randomized_controls import install

        install()
    options = sys.argv[1:]
    if os.environ.get('NEP_CONTROLLED_PROFILE', 'on') == 'off':
        filtered = []
        index = 0
        while index < len(options):
            flag = options[index]
            index += 1
            if flag in ('--profile', '--use-pytorch-profiler'):
                continue
            if flag in ('--profile-step-start', '--profile-step-end'):
                index += 1
                continue
            if flag == '--profile-ranks':
                while index < len(options) and not options[index].startswith('--'):
                    index += 1
                continue
            filtered.append(flag)
        sys.argv[1:] = filtered
        (root / f'effective-options-{os.environ["RANK"]}.json').write_text(
            json.dumps(dict(profile=False, options=filtered), indent=2) + '\n'
        )
        os.environ['MEGATRON_NONUNIFORM_EP_ENTRYPOINT'] = '1'
        print('CONTROLLED_PROFILE_DISABLED: no Nsight or event wrappers installed', flush=True)
        runpy.run_path(
            str(Path(os.environ['CODE_REPO']) / 'pretrain_hybrid.py'), run_name='__main__'
        )
    else:
        if os.environ.get('NEP_ACTUAL_A8B') == '1':
            install_delivery_markers()
        from pretrain_nep_cleanup_probe import main as profile

        profile()


if __name__ == '__main__':
    try:
        main()
    finally:
        if NATIVE_ITERATION_MS:
            (
                Path(os.environ['GRAPH_OUTPUT']) / f'native-timing-{os.environ["RANK"]}.json'
            ).write_text(json.dumps(NATIVE_ITERATION_MS) + '\n')
        if os.environ.get('NEP_WORK_ALLOCATION') == '1':
            import torch

            if torch.cuda.is_initialized():
                free, total = torch.cuda.mem_get_info()
                record = dict(
                    rank=int(os.environ['RANK']),
                    peak_allocated=torch.cuda.max_memory_allocated(),
                    peak_reserved=torch.cuda.max_memory_reserved(),
                    end_free=free,
                    total=total,
                )
                (Path(os.environ['GRAPH_OUTPUT']) / f'memory-{record["rank"]}.json').write_text(
                    json.dumps(record, indent=2) + '\n'
                )
