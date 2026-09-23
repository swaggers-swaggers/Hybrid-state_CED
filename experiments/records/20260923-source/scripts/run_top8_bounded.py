#!/usr/bin/env python3
"""One durable training worker, no log polling; hard timeout below eight hours."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
started=time.monotonic()
command=[sys.executable,str(ROOT/'scripts/train_top8_distill.py'),'--run','--stage','all','--tokens','500000',
         '--training-hours','6','--soft-hours','7.5','--evaluation-responses','64','--output',str(out/'experiment')]
(out/'launch.json').write_text(json.dumps(dict(command=command,started_unix=time.time(),hard_limit_seconds=28200),indent=2)+'\n')
with (out/'run.log').open('w') as log:
    worker=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    timed_out=False
    try:code=worker.wait(timeout=max(1,28200-(time.monotonic()-started)))
    except subprocess.TimeoutExpired:
        timed_out=True;os.killpg(worker.pid,signal.SIGTERM)
        try:code=worker.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(worker.pid,signal.SIGKILL);code=worker.wait()
status='HARD_TIME_LIMIT' if timed_out else ('COMPLETED' if code==0 else 'FAILED_OR_SOFT_TIME_LIMIT')
report=dict(status=status,returncode=code,elapsed_seconds=time.monotonic()-started,ended_unix=time.time())
tmp=out/'completion.partial';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(out/'completion.json')
sys.exit(0 if code==0 else 1)
