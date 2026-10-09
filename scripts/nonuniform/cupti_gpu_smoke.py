"""Check kernel/graph-node completeness of the native activity collector."""

import ctypes
import json
import os
from pathlib import Path

if os.environ.get('CUPTI_BACKEND_LIBRARY'):
    ctypes.CDLL(os.environ['CUPTI_BACKEND_LIBRARY'], mode=ctypes.RTLD_GLOBAL)
lib = ctypes.CDLL(os.environ['CUPTI_TRACE_LIBRARY'])
if os.environ.get('CUPTI_LATE_INIT') != '1':
    lib.gpu_trace_init()
    if os.environ.get('CUPTI_GPU_HES') == '1':
        assert ctypes.CDLL('libcuda.so.1').cuInit(0) == 0
        lib.gpu_trace_enable_hes()
    if hasattr(lib, 'gpu_trace_prepare'):
        lib.gpu_trace_prepare()
lib.gpu_trace_timestamp.restype = ctypes.c_uint64
lib.gpu_trace_range.argtypes = [ctypes.c_char_p, ctypes.c_uint64, ctypes.c_uint64]
lib.gpu_trace_stop.argtypes = [ctypes.c_char_p]

import torch

a = torch.randn(1024,1024,device='cuda')
b = torch.empty_like(a)
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    for _ in range(10): torch.mm(a,a,out=b)
s.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    for _ in range(10): torch.mm(a,a,out=b)
torch.cuda.synchronize()
if os.environ.get('CUPTI_LATE_INIT') == '1':
    assert os.environ.get('CUPTI_GPU_HES') != '1'
    lib.gpu_trace_init()
    lib.gpu_trace_prepare()
graph_events = None
if os.environ.get('CUPTI_GRAPH_EVENTS') == '1':
    from cupti_capture_adapter import GraphEvents

    graph_events = GraphEvents(torch, lib, capacity=4)
    graph_events.start()
lib.gpu_trace_start()
start = lib.gpu_trace_timestamp()
b.copy_(a)
cudart = ctypes.CDLL('libcudart.so.13')
cudart.cudaMemsetAsync.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p]
assert cudart.cudaMemsetAsync(b.data_ptr(), 0, b.numel()*b.element_size(),
                             torch.cuda.current_stream().cuda_stream) == 0
for _ in range(10): torch.mm(a,a,out=b)
g.replay()
g.replay()
torch.cuda.synchronize()
if graph_events is not None:
    graph_events.finish()
lib.gpu_trace_range(b'iteration_10',start,lib.gpu_trace_timestamp())
path = Path(os.environ['PROFILER_INSPECT_ROOT'])/'gpu_smoke.json'
lib.gpu_trace_stop(str(path).encode())
d = json.loads(path.read_text())
assert d['dropped_records']==0 and d['invalid_records']==0,d
if os.environ.get('CUPTI_MEMORY_ACTIVITY') != '0':
    for category in ('gpu_memcpy', 'gpu_memset'):
        assert any(e['cat']==category and e['args']['bytes']==b.numel()*b.element_size()
                   for e in d['traceEvents']), category
if os.environ.get('CUPTI_EAGER_ONLY') == '1':
    full_hes = (os.environ.get('CUPTI_GPU_HES') == '1'
                and os.environ.get('CUPTI_LATE_GRAPH_ACTIVITY') == '1')
    expected_kernels = 30 if full_hes else 10
    assert sum(e['cat']=='kernel' for e in d['traceEvents'])==expected_kernels
    assert sum(e['cat']=='graph_span' for e in d['traceEvents'])==(2 if graph_events or os.environ.get('CUPTI_MINIMAL_FULL') == '1' else 0)
    print(f'PASS: {expected_kernels} kernels; graph body capture={full_hes}; '
          'graph-envelope count verified; zero dropped/invalid records')
elif os.environ.get('CUPTI_WHOLE_GRAPH') == '1':
    assert sum(e['cat']=='kernel' for e in d['traceEvents'])==10
    assert sum(e['cat']=='graph_span' for e in d['traceEvents'])==2
    print('PASS: 10 eager kernels and2 whole-graph spans; zero dropped/invalid records')
else:
    assert sum(e['cat']=='kernel' for e in d['traceEvents'])==30
    print('PASS: 30 kernels, including20 graph-node kernels; zero dropped/invalid records')
