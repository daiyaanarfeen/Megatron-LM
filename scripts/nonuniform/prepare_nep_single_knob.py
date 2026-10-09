"""Freeze trusted runtime and benchmark launchers for the sensitivity experiment."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path


REPO = Path('/home/darfeen/Megatron-LM')
TRUSTED = REPO / '.worktrees/nep-sync-cpu-20260914'
SNAPSHOT = Path('/home/darfeen/Megatron-LM-nep-sensitivity-accepted-20260914')
BENCH = SNAPSHOT / 'sensitivity_harness'
EXPECTED = '653a7efd3da53c0ace84c9e32d65eb2faa1f9b17f44b1daf43ace136a3eae0c7'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert digest(TRUSTED / 'megatron/core/distributed/nonuniform_ep.py') == EXPECTED
    assert not SNAPSHOT.exists(), 'Never overwrite an experiment snapshot'
    commit = subprocess.check_output(['git', '-C', str(TRUSTED), 'rev-parse', 'HEAD'], text=True).strip()
    assert commit == '93d313df4d5588795ea628592668b188ed352a66'
    assert not subprocess.check_output(['git', '-C', str(TRUSTED), 'status', '--porcelain'])
    subprocess.run(['git', '-C', str(TRUSTED), 'worktree', 'add', '--detach', str(SNAPSHOT), commit], check=True)
    tracked = subprocess.check_output(['git', '-C', str(TRUSTED), 'ls-files', '-z'])
    subprocess.run(['rsync', '-a', '--from0', '--files-from=-', str(TRUSTED) + '/', str(SNAPSHOT) + '/'], input=tracked, check=True)
    for helper in (TRUSTED / 'megatron/core/datasets').glob('helpers_cpp*.so'):
        shutil.copy2(helper, SNAPSHOT / helper.relative_to(TRUSTED))
    scripts = BENCH / 'scripts/nonuniform'
    scripts.mkdir(parents=True)
    names = [
        'run_lyris_smallest_soft_clique.sh', 'run_lyris_training_scripts_nep_ab.sh',
        'run_lyris_soft_clique_rank.sh', 'run_lyris_training_scripts_nep_rank.sh',
        'run_lyris_a3b_direct_rank.sh', 'training_scripts_nep_matrix.py',
        'nep_nvl_placement.py', 'pretrain_profile_activity_probe.py',
        'cuda_event_capture_adapter.py', 'run_nep_single_knob_rank.py',
    ]
    originals = {}
    for name in names:
        source = REPO / 'scripts/nonuniform' / name
        originals[name] = digest(source)
        text = source.read_text()
        for runtime in (str(TRUSTED), '/home/darfeen/Megatron-LM-nep-main-20260817'):
            text = text.replace(runtime, '@@RUNTIME@@')
        text = text.replace(str(REPO), str(BENCH)).replace('@@RUNTIME@@', str(SNAPSHOT))
        if name == 'run_lyris_soft_clique_rank.sh':
            old = 'exec bash "$(dirname "${BASH_SOURCE[0]}")/run_lyris_training_scripts_nep_rank.sh"'
            assert old in text
            text = text.replace(old, 'exec python "$(dirname "${BASH_SOURCE[0]}")/run_nep_single_knob_rank.py"')
        (scripts / name).write_text(text)
    shutil.copytree(REPO / 'examples/training_scripts', BENCH / 'examples/training_scripts')
    files = {}
    for relative in tracked.decode().split('\0'):
        if relative and (SNAPSHOT / relative).is_file():
            files[relative] = digest(SNAPSHOT / relative)
    harness = {str(p.relative_to(BENCH)): digest(p) for p in BENCH.rglob('*') if p.is_file()}
    built = {str(p.relative_to(SNAPSHOT)): digest(p)
             for p in (SNAPSHOT / 'megatron/core/datasets').glob('helpers_cpp*.so')}
    assert built, 'The trusted dataset extension must be included'
    manifest = dict(commit=commit, source=str(TRUSTED), snapshot=str(SNAPSHOT),
                    built_extension_sha256=built,
                    runtime_sha256=files, original_harness_sha256=originals,
                    frozen_harness_sha256=harness)
    (SNAPSHOT / 'sensitivity_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(snapshot=str(SNAPSHOT), benchmark=str(BENCH), commit=commit,
                          nonuniform_ep_sha256=files['megatron/core/distributed/nonuniform_ep.py']), indent=2))


if __name__ == '__main__':
    main()
