#!/usr/bin/env python3
"""Run the repaired experiment once; keep legacy outputs and all subprocess logs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/phase1_corrected'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', CED_RUN_MODEL_TESTS='1')
    stages = []
    def call(name, args, allowed=(0,)):
        start = time.time()
        with (OUT / f'{name}.log').open('w') as log:
            code = subprocess.call([sys.executable, *args], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        stages.append({'stage': name, 'exit_code': code, 'seconds': time.time()-start})
        (OUT/'run_stages.json').write_text(json.dumps(stages, indent=2)+'\n')
        if code not in allowed:
            raise RuntimeError(f'{name} failed with exit {code}; see {OUT/name}.log')
    # Confirm existing splits without downloading or modifying their contents.
    manifest = json.loads((ROOT/'data/phase1_wikitext103/manifest.json').read_text())
    for split in manifest['splits'].values():
        path = ROOT/'data/phase1_wikitext103'/split['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == split['sha256']
    call('preflight_tests', ['-m','unittest','discover','-s','tests','-v'])
    for layer in (19,15,23):
        arguments = ['--target-layer',str(layer),'--checkpoint-path',f'checkpoints/phase1_corrected/layer{layer}_kv_probes.pt','--output-dir',f'results/phase1_corrected/layer{layer}']
        call(f'train_layer{layer}', ['scripts/train_kv_probe.py', *arguments])
        call(f'eval_layer{layer}', ['scripts/evaluate_kv_probe.py', *arguments], allowed=(0,2))
    call('aggregate', ['scripts/aggregate_phase1.py','--results-dir','results/phase1_corrected'], allowed=(0,2))
    pilot = json.loads((ROOT/'configs/kv_continuation_pilot.json').read_text())
    pilot.update({'seed': 20260918, 'checkpoint_dir': 'checkpoints/phase1_corrected',
        'require_corrected_checkpoint': True, 'normalized_original_projection': True,
        'exclude_windows_from': 'results/kv_continuation_pilot/results.json'})
    for layer in (15,19,23):
        evaluation = json.loads((OUT/f'layer{layer}/phase1_results.json').read_text())
        pilot['selected_methods'][str(layer)] = evaluation['selected_method']
    pilot['scope'] = 'Corrected target and normalized original-projection baseline; same fixed pilot thresholds; no overlap with previous pilot windows; reused Phase 1 test corpus, not new independent data; no training/tuning on continuation results.'
    config = OUT/'continuation_config.json'
    config.write_text(json.dumps(pilot, ensure_ascii=False, indent=2)+'\n')
    call('continuation', ['scripts/run_kv_continuation_pilot.py','--config',str(config),'--output-dir',str(OUT/'continuation')])
    call('final_tests', ['-m','unittest','discover','-s','tests','-v'])
    print(json.dumps({'complete': True, 'output': str(OUT), 'stages': stages}, ensure_ascii=False))


if __name__ == '__main__':
    main()
