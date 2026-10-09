#!/usr/bin/env python3
"""Opt-in wall/thread-CPU measurements; no CUDA synchronization or model changes."""

import atexit
import ctypes
import functools
import importlib
import inspect
import json
import os
import resource
import runpy
import sys
import threading
import time
from pathlib import Path

ROWS = []
ITERATION = -1
START = int(os.environ.get('NEP_CPU_DIAG_START', '10'))
END = int(os.environ.get('NEP_CPU_DIAG_END', '22'))
CPU_ID = lambda: -1


def measured(function, label):
    """Preserve return/exception behavior and measure the calling thread only."""
    parameters = list(inspect.signature(function).parameters)
    split_index = parameters.index('m_splits') if 'm_splits' in parameters else None

    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        if not START <= ITERATION <= END:
            return function(*args, **kwargs)
        splits = kwargs.get('m_splits')
        if splits is None and split_index is not None and split_index < len(args):
            splits = args[split_index]
        split_info = list(splits) if isinstance(splits, (list, tuple)) and all(
            isinstance(x, int) for x in splits) else None
        tid = threading.get_native_id()
        processor = CPU_ID()
        before = resource.getrusage(resource.RUSAGE_THREAD)
        cpu_start = time.thread_time_ns()
        wall_start = time.time_ns()
        monotonic_start = time.perf_counter_ns()
        try:
            return function(*args, **kwargs)
        finally:
            wall_elapsed = time.perf_counter_ns() - monotonic_start
            cpu_elapsed = time.thread_time_ns() - cpu_start
            after = resource.getrusage(resource.RUSAGE_THREAD)
            if (os.environ.get('NEP_CPU_DIAG_SPLITS') == '1'
                    and ITERATION == START + 9 and label == 'forward'
                    and getattr(getattr(splits, 'device', None), 'type', None) == 'cpu'):
                # Snapshot after the original call, without reading a CUDA tensor.
                split_info = splits.tolist()
            ROWS.append((ITERATION, label, tid, wall_start, wall_elapsed, cpu_elapsed,
                         after.ru_nvcsw - before.ru_nvcsw,
                         after.ru_nivcsw - before.ru_nivcsw, split_info,
                         processor, CPU_ID()))

    return wrapped


def main():
    global CPU_ID
    CPU_ID = ctypes.CDLL(None).sched_getcpu
    CPU_ID.restype = ctypes.c_int
    CPU_ID.argtypes = []
    repo = Path(os.environ['REPO_DIR'])
    routing = os.environ.get('NEP_CPU_DIAG_ROUTING', 'existing')
    if routing not in ('existing', 'uniform'):
        raise ValueError(f'Unsupported diagnostic routing override: {routing}')
    if routing == 'uniform':
        sys.argv = [arg for arg in sys.argv if arg != '--moe-router-force-load-balancing']
        sys.argv.append('--moe-router-force-uniform-routing')
    sys.path.insert(0, str(repo))
    grouped = importlib.import_module('transformer_engine.pytorch.module.grouped_linear')
    training = importlib.import_module('megatron.training.training')
    # The NEP shim execv's this file. Run its final target directly so hooks survive.
    target = repo / 'pretrain_hybrid.py'
    os.environ['MEGATRON_NONUNIFORM_EP_ENTRYPOINT'] = '1'
    output = Path(os.environ['NEP_CPU_DIAG_OUTPUT']) / os.environ['NONUNIFORM_MODE']
    output.mkdir(parents=True, exist_ok=True)
    metadata = {'rank': int(os.environ['RANK']), 'host': os.uname().nodename,
                'affinity': sorted(os.sched_getaffinity(0)),
                'slurm_cpus_per_task': os.environ.get('SLURM_CPUS_PER_TASK'),
                'slurm_cpu_bind': os.environ.get('SLURM_CPU_BIND'),
                'omp_num_threads': os.environ.get('OMP_NUM_THREADS'),
                'grouped_source': inspect.getsourcefile(grouped),
                'diagnostic_version': 2,
                'routing_override': routing,
                'columns': ['iteration_zero_based', 'operation', 'tid', 'unix_start_ns',
                            'wall_ns', 'thread_cpu_ns', 'voluntary_switches',
                            'involuntary_switches', 'm_splits', 'start_cpu', 'end_cpu']}
    for method in ('forward', 'backward'):
        original = getattr(grouped._GroupedLinear, method)
        metadata[f'{method}_signature'] = str(inspect.signature(original))
        setattr(grouped._GroupedLinear, method, staticmethod(measured(original, method)))
    original_step = training.train_step
    measured_step = measured(original_step, 'train_step')

    @functools.wraps(original_step)
    def step(*args, **kwargs):
        global ITERATION
        from megatron.training import get_args
        ITERATION = get_args().curr_iteration
        result = measured_step(*args, **kwargs)
        if ITERATION == END:
            save()
        return result

    training.train_step = step

    def save():
        (output / f'rank-{metadata["rank"]}.json').write_text(
            json.dumps({'metadata': metadata, 'rows': ROWS}) + '\n')
        print(f'[nep-cpu-diagnostic] saved rank={metadata["rank"]} records={len(ROWS)}', flush=True)

    atexit.register(save)
    print('[nep-cpu-diagnostic]', json.dumps(metadata), flush=True)
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name='__main__')


if __name__ == '__main__':
    main()
