#!/usr/bin/env python3
"""Top8 pair-distillation entry. Default is a plan; verification never updates weights."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ced_distill.protocol import plan,SEMANTICS
from ced_training.protocol import sha256

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/qwen35_08b_top8_distill.json')
    p.add_argument('--stage',choices=('all','modules','confidence','calibrate','evaluate'),default='all')
    p.add_argument('--checkpoint',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--start',type=int,default=0);p.add_argument('--tokens',type=int,default=1_000_000)
    p.add_argument('--training-hours',type=float,help='Size data using first 1024 positions, stop only at a complete pair boundary')
    p.add_argument('--evaluation-responses',type=int,help='Fixed response limit for each held-out and gate split')
    p.add_argument('--soft-hours',type=float,help='Whole worker time limit; completed-stage checkpoints survive')
    p.add_argument('--smoke',action='store_true',help='Separate bounded experiment; still requires --run')
    mode=p.add_mutually_exclusive_group();mode.add_argument('--run',action='store_true');mode.add_argument('--verify',action='store_true');mode.add_argument('--plan',action='store_true')
    args=p.parse_args();c=json.loads(args.config.read_text());count=64 if args.smoke else args.tokens
    if args.smoke:c={**c,'gate_epochs':1,'eval_prompts':2,'decode_lengths':[8]}
    experiment=plan(ROOT,c,args.start,count)
    experiment['requested_stage']=args.stage
    if args.training_hours is not None and not 0<args.training_hours<=6:p.error('training-hours must be in (0,6]')
    if args.soft_hours is not None and not 0<args.soft_hours<=7.5:p.error('soft-hours must be in (0,7.5]')
    if args.evaluation_responses is not None and args.evaluation_responses<1:p.error('evaluation-responses must be positive')
    experiment['execution_limits']={'training_hours':args.training_hours,'evaluation_responses':args.evaluation_responses,'soft_hours':args.soft_hours}
    if args.stage not in ('all','modules'):
        experiment['interval']='Inherited from the preceding checkpoint at execution'
    if not args.run and not args.verify:
        print(json.dumps(experiment,ensure_ascii=False,indent=2));return
    if args.output is None:p.error('--run/--verify requires a fresh output directory')
    if args.stage not in ('all','modules') and not args.checkpoint:p.error('This stage requires its preceding checkpoint')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    import torch
    from ced_distill.data import SparseData
    from ced_distill.runtime import Runtime
    from ced_training.topk_data import check_storage
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    def guard(size):check_storage(output,size,c['maximum_artifact_bytes'],c['minimum_free_bytes'])
    def write(name,obj):
        body=json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n';guard(len(body.encode()))
        (output/name).write_text(body)
    started=time.monotonic();write('plan.json',experiment)
    if args.soft_hours is not None:
        import signal
        def deadline(signum,frame):raise TimeoutError('Worker soft time budget reached; do not launch additional stages')
        signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,args.soft_hours*3600)
    hashes={str(f.relative_to(ROOT)):sha256(f) for d in ('ced_distill','ced_training','exit_cost') for f in (ROOT/d).glob('*.py')}
    hashes['scripts/train_top8_distill.py']=sha256(Path(__file__))
    write('source_hashes.json',hashes)
    try:
        data=SparseData(ROOT,c);rt=Runtime(ROOT,c,data,args.checkpoint)
        write('source_checkpoint.json',{'path':rt.source,'sha256':rt.source_sha256,'all_modules_exactly_loaded':rt.initialization})
        if args.verify:
            from ced_distill.verify import verify
            report=verify(rt,data)
            write('verification.json',report)
            print(json.dumps(report,indent=2));return
        expected={'confidence':'modules','calibrate':'confidence','evaluate':'calibrated'}
        if args.stage in expected and (rt.loaded is None or rt.loaded['stage']!=expected[args.stage]):raise ValueError('Wrong checkpoint stage')
        if args.stage in ('all','modules'):
            if args.start and (rt.loaded is None or rt.loaded['stage']!='modules' or rt.loaded['interval'][1]!=args.start):raise ValueError('Continuation must start at previous module interval end')
            if not args.start and rt.loaded is not None:raise ValueError('Do not restart the same data interval from a Top8 checkpoint')
        stages=['modules','confidence','calibrate','evaluate'] if args.stage=='all' else [args.stage]
        report={'status':'RUNNING','semantics':SEMANTICS,'stages':{}}
        interval=[args.start,args.start+count] if args.stage in ('all','modules') else rt.loaded['interval']
        write('plan.json',{**experiment,'interval':interval})
        def save(stage):
            guard(40_000_000)
            state={'semantics':SEMANTICS,'stage':stage,'manifest_sha256':data.digest,'config':c,'interval':interval,
                   'model_weights_sha256':data.manifest['model_weights_sha256'],'source_sha256':rt.source_sha256,
                   'modules':rt.state(),'threshold':rt.threshold,'smoke':args.smoke,'execution_limits':experiment['execution_limits']}
            temp=output/(stage+'.partial');torch.save(state,temp);temp.replace(output/(stage+'.pt'))
        limit=2 if args.smoke else args.evaluation_responses
        for stage in stages:
            if any(sha256(ROOT/name)!=digest for name,digest in hashes.items()):raise RuntimeError('Experiment code changed')
            step_start=time.monotonic()
            if stage=='modules':
                from ced_distill.training import module_pass
                from ced_distill.budget import TokenBudget
                budget=TokenBudget(count,args.training_hours*3600) if args.training_hours is not None else None
                result=module_pass(rt,data,'train',args.start,count,output,train=True,storage_guard=guard,budget=budget)
                interval=[args.start,args.start+result['positions']]
                save('modules');write('modules.json',result)
                write('plan.json',{**experiment,'interval':interval,'requested_maximum_tokens':count})
                result['dev']=module_pass(rt,data,'dev',0,None,train=False,max_responses=limit)
            elif stage=='confidence':
                from ced_distill.training import train_confidence
                result=train_confidence(rt,data,limit);save('confidence')
            elif stage=='calibrate':
                from ced_distill.evaluation import calibrate_gate
                result=calibrate_gate(rt,data,limit);save('calibrated')
            else:
                from ced_distill.evaluation import evaluate
                result=evaluate(rt,data,limit)
            result['elapsed_seconds']=time.monotonic()-step_start;write(stage+'.json',result);report['stages'][stage]=result
        report.update(status='COMPLETED',elapsed_seconds=time.monotonic()-started,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
        write('result.json',report)
        print(json.dumps({'status':report['status'],'output':str(output),'elapsed_seconds':report['elapsed_seconds']},indent=2))
    except BaseException:
        if args.soft_hours is not None:signal.setitimer(signal.ITIMER_REAL,0)
        import traceback
        write('failure.json',{'traceback':traceback.format_exc(),'elapsed_seconds':time.monotonic()-started})
        raise
if __name__=='__main__':main()
