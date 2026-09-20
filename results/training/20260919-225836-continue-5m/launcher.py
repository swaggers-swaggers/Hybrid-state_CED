"""Execute the five explicitly authorized 1M chunks, with completion-only output."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
ROOT = Path('/home/liu/CED')
SOURCE = ROOT / 'results/training/20260919-223920-fresh-chunk01-1m'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)

def main():
    assert sys.argv[1:] == ['--run'], 'Explicit --run required'
    previous_record = json.loads((SOURCE / 'execution.json').read_text())
    assert previous_record['status'] == 'COMPLETED'
    assert previous_record['chunk_index'] == 0
    warm = SOURCE / 'modules/modules.pt'
    assert warm.is_file()
    output = ROOT / 'results/training' / f'{datetime.now():%Y%m%d-%H%M%S}-continue-5m'
    output.mkdir(exist_ok=False)
    (output / 'launcher.py').write_bytes(Path(__file__).read_bytes())
    files = [p for name in ('scripts','ced_training','exit_cost') for p in (ROOT / name).glob('*.py')]
    files += [ROOT / 'configs/qwen35_08b_train.json']
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in files}
    save(output / 'source_hashes.json', hashes)
    record = {'status':'RUNNING', 'output':str(output), 'source_run':str(SOURCE),
              'requested_additional_effective_targets':5_000_000,
              'interval':[1_000_000,6_000_000], 'chunk_indices':[1,2,3,4,5],
              'inspection_policy':'Do not inspect metrics or tune during execution; analyze all five chunks after completion.',
              'runs':[]}
    start = time.monotonic()
    save(output / 'execution.json',record)
    try:
        for index in range(1,6):
            if any(sha(ROOT / name) != digest for name,digest in hashes.items()):
                raise RuntimeError('Source changed; stop before the next chunk')
            run = output / f'chunk{index+1:02d}-1m'
            command = [sys.executable,str(ROOT/'scripts/run_incremental_training.py'),
                       '--run','--chunk-index',str(index),'--warm-start',str(warm),'--output',str(run)]
            with (output / f'chunk{index+1:02d}.log').open('w') as log:
                result = subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'Chunk {index+1} failed with exit code {result.returncode}; stop dependent chunks')
            # Completion checks only: no reading metrics or altering hyperparameters.
            state = json.loads((run/'execution.json').read_text())
            if state['status'] != 'COMPLETED' or len(state['stages']) != 4 or any(s['returncode'] for s in state['stages']):
                raise RuntimeError('Chunk did not finish its four required stages')
            record['runs'].append({'chunk_index':index,'path':str(run),'warm_start':str(warm),
                                   'warm_start_sha256':sha(warm),'wall_seconds':state['wall_seconds']})
            save(output/'execution.json',record)
            warm = run/'modules/modules.pt'
        record['status'] = 'COMPLETED'
        record['completed_additional_effective_targets'] = 5_000_000
        record['remaining_fresh_effective_targets'] = 4_000_000
        record['final_modules_checkpoint'] = str(warm)
    except BaseException as exc:
        record['status'],record['error'] = 'FAILED',str(exc)
        raise
    finally:
        record['wall_seconds'] = time.monotonic()-start
        save(output/'execution.json',record)
        print(json.dumps(record,ensure_ascii=False,indent=2),flush=True)

if __name__ == '__main__':
    main()
