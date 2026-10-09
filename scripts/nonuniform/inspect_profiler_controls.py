"""Inspect installed profiler controls and headers in the benchmark container."""

import os
import shutil
import subprocess
from pathlib import Path

import torch

root=Path(os.environ['PROFILER_INSPECT_ROOT'])
root.mkdir(parents=True,exist_ok=True)
print('TORCH',torch.__version__,torch._C._profiler._ExperimentalConfig.__init__.__doc__,flush=True)
nsys=Path(shutil.which('nsys')).resolve()
print('NSYS',nsys,flush=True)
for p in nsys.parent.glob('libToolsInjection*.so'):
    output=subprocess.run(['strings',str(p)],capture_output=True,text=True,check=True).stdout
    lines=[s for s in output.splitlines() if s.startswith(('Cupti','CUPTI_','EnableCuda','DisableCuda','CudaTrace'))]
    print(p.name,'\n'.join(lines),flush=True)
for p in Path('/usr/local').glob('cuda*/extras/CUPTI/include/cupti_activity.h'):
    print('HEADER',p,flush=True)
    shutil.copy(p,root/p.name)
for p in Path('/usr/local').glob('cuda*/include/cupti_activity.h'):
    print('HEADER',p,flush=True)
    shutil.copy(p,root/p.name)
print('CUPTI libraries',list(Path('/usr/local').glob('cuda*/**/libcupti.so*')),flush=True)
