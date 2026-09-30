"""Small regression gate; profiling-off runs are explicitly user-authorized."""

import json
import os
import runpy
import sys
from pathlib import Path


def main():
    root = Path(os.environ['GRAPH_OUTPUT'])
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
        from pretrain_nep_cleanup_probe import main as profile

        profile()


if __name__ == '__main__':
    main()
