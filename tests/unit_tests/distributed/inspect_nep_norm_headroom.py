"""Archive installed normalization sources and verify the native SM-margin knob."""

import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path

import torch
import transformer_engine as te
import transformer_engine.pytorch as tep


def main():
    root = Path(os.environ['NORM_SOURCE_OUTPUT'])
    modules = (
        'transformer_engine.pytorch.module.rmsnorm',
        'transformer_engine.pytorch.module.layernorm_linear',
        'transformer_engine.pytorch.ops.basic.rmsnorm',
    )
    sources = {}
    for name in modules:
        module = importlib.import_module(name)
        source = inspect.getsource(module)
        (root / f'{name}.py').write_text(source)
        sources[name] = dict(
            path=module.__file__, sha256=hashlib.sha256(source.encode()).hexdigest()
        )
    margins = {}
    for margin in (32, 64):
        os.environ['NVTE_BWD_LAYERNORM_SM_MARGIN'] = str(margin)
        norm = tep.RMSNorm(4608, device='cuda', params_dtype=torch.bfloat16)
        values = {k: v for k, v in vars(norm).items() if 'margin' in k}
        assert values, 'Installed RMSNorm exposes no SM margin: inspect before continuing'
        assert any(
            v == margin or isinstance(v, dict) and v.get('backward', v.get('bwd')) == margin
            for v in values.values()
        ), values
        margins[margin] = values
        del norm
    props = torch.cuda.get_device_properties(0)
    result = dict(
        torch_version=torch.__version__,
        te_version=te.__version__,
        gpu=props.name,
        multiprocessor_count=props.multi_processor_count,
        margins=margins,
        sources=sources,
    )
    (root / 'installed.json').write_text(json.dumps(result, indent=2) + '\n')
    print('INSTALLED_NORM_HEADROOM', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
