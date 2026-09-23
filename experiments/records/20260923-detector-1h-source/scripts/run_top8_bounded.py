#!/usr/bin/env python3
"""One durable training worker, no log polling, with an explicit hard deadline."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
p.add_argument('--tokens',type=int,default=500000)
p.add_argument('--training-hours',type=float,default=6.)
p.add_argument('--soft-hours',type=float,default=7.5)
p.add_argument('--hard-seconds',type=float,default=28200.)
p.add_argument('--evaluation-responses',type=int,default=64)
args=p.parse_args()
if not 0<args.training_hours<args.soft_hours or not args.soft_hours*3600<args.hard_seconds<=28800:
    p.error('Require training < worker soft limit < hard limit <= 8 hours')
out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
started=time.monotonic()
command=[sys.executable,str(ROOT/'scripts/train_top8_distill.py'),'--run','--stage','all','--tokens',str(args.tokens),
         '--training-hours',str(args.training_hours),'--soft-hours',str(args.soft_hours),
         '--evaluation-responses',str(args.evaluation_responses),'--output',str(out/'experiment')]
(out/'launch.json').write_text(json.dumps(dict(command=command,started_unix=time.time(),hard_limit_seconds=args.hard_seconds),indent=2)+'\n')
with (out/'run.log').open('w') as log:
    worker=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    timed_out=False
    try:code=worker.wait(timeout=max(1,args.hard_seconds-(time.monotonic()-started)))
    except subprocess.TimeoutExpired:
        timed_out=True;os.killpg(worker.pid,signal.SIGTERM)
        try:code=worker.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(worker.pid,signal.SIGKILL);code=worker.wait()
status='HARD_TIME_LIMIT' if timed_out else ('COMPLETED' if code==0 else 'FAILED_OR_SOFT_TIME_LIMIT')
report=dict(status=status,returncode=code,elapsed_seconds=time.monotonic()-started,ended_unix=time.time())
tmp=out/'completion.partial';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(out/'completion.json')
sys.exit(0 if code==0 else 1)
