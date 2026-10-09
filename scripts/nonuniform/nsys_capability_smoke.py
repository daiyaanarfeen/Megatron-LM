"""Check Nsight CUDA tracing support without touching the training workload."""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def workload():
    import torch

    a = torch.randn(1024, 1024, device='cuda')
    out = torch.empty_like(a)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(10):
            torch.mm(a, a, out=out)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            torch.mm(a, a, out=out)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(10):
        torch.mm(a, a, out=out)
    for _ in range(2):
        graph.replay()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print('Expected: 30 kernels at node granularity, or 10 kernels + 2 graph executions.')


def main():
    root = Path(os.environ['NSYS_SMOKE_ROOT'])
    root.mkdir(parents=True, exist_ok=True)
    nsys = shutil.which('nsys')
    print(f'nsys={nsys}', flush=True)
    if nsys is None:
        raise SystemExit('Nsight Systems is not installed in this container')
    for cmd, name in [([nsys, '--version'], 'version'), ([nsys, 'profile', '--help'], 'help'), (['nvidia-smi'], 'gpu')]:
        output = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True).stdout
        (root / f'{name}.txt').write_text(output)
        if name != 'help':
            print(output, flush=True)
    help_text = (root / 'help.txt').read_text()
    traces = ['cuda'] + (['cuda-hw'] if 'cuda-hw' in help_text else [])
    for trace in traces:
        base = root / trace
        cmd = [nsys, 'profile', f'--trace={trace}', '--sample=none', '--cpuctxsw=none', '--cuda-graph-trace=node', '--cuda-event-trace=false', '--capture-range=cudaProfilerApi', '--capture-range-end=stop', '--force-overwrite=true', f'--output={base}', sys.executable, __file__, '--workload']
        print('COMMAND', cmd, flush=True)
        with (root / f'{trace}.log').open('w') as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=90)
        print('RESULT', trace, result.returncode, (root / f'{trace}.log').read_text(), flush=True)
        report = base.with_suffix('.nsys-rep')
        if result.returncode == 0 and report.exists():
            exported = subprocess.run([nsys, 'export', '--type=sqlite', '--force-overwrite=true', f'--output={base}.sqlite', str(report)], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
            print('EXPORT', exported.returncode, exported.stdout, flush=True)


if __name__ == '__main__':
    if '--workload' in sys.argv:
        workload()
    else:
        main()
