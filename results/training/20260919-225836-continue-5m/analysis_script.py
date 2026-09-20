"""Post-completion CPU audit and comparison; never used while a batch is running."""
from pathlib import Path
import hashlib,json,statistics,sys
import torch
ROOT=Path('/home/liu/CED')
STAGE=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
batch=Path(sys.argv[1]).resolve()
execution=read(batch/'execution.json')
assert execution['status']=='COMPLETED' and len(execution['runs'])==5
baseline=Path(execution['source_run'])
paths=[baseline]+[Path(r['path']) for r in execution['runs']]
summary={'execution':execution,'runs':[],'audit':[]}
prior=None
prior_path=None
for p in paths:
    r=read(p/'execution.json');m=read(p/'modules/result.json');c=read(p/'calibrate/calibration.json');e=read(p/'evaluate/result.json')
    assert r['status']=='COMPLETED' and all(s['returncode']==0 for s in r['stages'])
    assert m['supervised_tokens']==1000000
    state=torch.load(p/'modules/modules.pt',map_location='cpu',weights_only=True)
    chunk=state['chunk'];i=chunk['chunk_index']
    assert chunk==read(p/'modules/chunk.json')
    assert chunk['start_effective_target']==i*1000000 and chunk['end_effective_target_exclusive']==(i+1)*1000000
    assert sum(end-start for _,start,end in chunk['spans'])==1000000
    assert state['config']['eval_every_steps']==1000000
    train=[json.loads(line) for line in (p/'modules/train.jsonl').read_text().splitlines()]
    dev_steps=[row['step'] for row in train if 'dev' in row]
    assert dev_steps==[m['steps']]
    if prior is not None:
        assert i==prior['chunk']['chunk_index']+1
        assert state['scales']==prior['scales']
        assert read(p/'modules/initial_dev.json')==prior['extra']['dev']
        assert state['incremental_manifest_sha256']==prior['incremental_manifest_sha256']
        assert state['config']['incremental']['source_checkpoint_sha256']==sha(prior_path/'modules/modules.pt')
        assert Path(r['warm_start'])==prior_path/'modules/modules.pt'
        for name in ('gate','dev','calibration','test'):
            assert state['split_indices'][name]==prior['split_indices'][name]
        a=prior['chunk']['spans'][-1];b=chunk['spans'][0]
        assert (a[0]==b[0] and a[2]==b[1]) or (b[0]==a[0]+1 and a[2]==255 and b[1]==0)
    weights={}
    for name in ('modules/modules.pt','modules/last_modules.pt','confidence/confidence.pt','calibrate/calibrated.pt'):
        s=torch.load(p/name,map_location='cpu',weights_only=True)
        assert s['chunk']==chunk
        params=[v for module in s['modules'].values() for v in module.values()]
        assert all(torch.isfinite(v).all().item() for v in params)
        weights[name]={'sha256':sha(p/name),'finite':True,'parameters':sum(v.numel() for v in params),'selection':s['extra']}
    source_hashes=read(p/'source_hashes.json')
    assert all(sha(ROOT/name)==digest for name,digest in source_hashes.items())
    continuations=read(p/'evaluate/continuations.json');gens=read(p/'evaluate/generations.json')
    contexts={}
    for context in (256,2048):
        cases={}
        for case in dict.fromkeys(x['case'] for x in continuations):
            xs=[x for x in continuations if x['context']==context and x['case']==case]
            cases[case]={key:statistics.mean(x[key] for x in xs) if all(x[key] is not None for x in xs) else None
                         for key in ('mean_kl','resume_mean_kl','exit_rate','exit_disagreement','top1_agreement','nll','teacher_nll')}
        xs=[x for x in gens if x['context']==context]
        contexts[str(context)]={'state_cases':cases,'generation':{
            'pairs':len(xs),'identical_pairs':sum(x['full']['ids']==x['adaptive']['ids'] for x in xs),
            'adaptive_exits':sum(x['adaptive']['exits'] for x in xs),
            'full_ms':statistics.mean(x['full']['ms_per_token'] for x in xs),
            'adaptive_ms':statistics.mean(x['adaptive']['ms_per_token'] for x in xs),
            'paired_median_speedup':statistics.median(x['speedup'] for x in xs)}}
    summary['runs'].append({'path':str(p),'cumulative_fresh_targets':(i+1)*1000000,
        'wall_seconds':r['wall_seconds'],'stages':r['stages'],'steps':m['steps'],'selected_step':state['extra']['step'],
        'supervised_tokens':m['supervised_tokens'],'chunk':{k:v for k,v in chunk.items() if k!='spans'},
        'peak_allocated_mib':max(read(p/stage/'result.json').get('peak_allocated_mib',0) for stage in ('modules','confidence','calibrate','evaluate')),
        'static_test':e['static_test'],'confidence_test':e['confidence_test'],'gate_enabled':e['gate_enabled'],
        'calibration':c,'contexts':contexts,'runtime_validation':e['runtime_validation'],'checkpoint_audit':weights})
    summary['audit'].append({'path':str(p),'status':'PASS','dev_evaluation_steps':dev_steps,'interval':[(i)*1000000,(i+1)*1000000]})
    prior,prior_path=state,p
before=summary['runs'][0]['static_test'];after=summary['runs'][-1]['static_test']
summary['change']={'agreement_percentage_points':100*(after['top1_agreement']-before['top1_agreement']),
                  'kv_reduction_pct':100*(1-after['kv']/before['kv']),
                  'ce_reduction_pct':100*(1-after['ce']/before['ce']),
                  'kd_reduction_pct':100*(1-after['kd']/before['kd'])}
summary['limits']=['Previously explored English WikiText diagnostic set, not new task accuracy or a generalization benchmark.',
                  'Each chunk inherits selected M12 and KV projectors; optimizer and confidence training are reset per existing protocol.',
                  'Unselected forced exits do not test the selective easy-token GDN-hold assumption.',
                  'Fallback-only equal generation does not demonstrate early-exit quality.']
(STAGE/'five_chunks_analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'execution':execution,'change':summary['change'],'runs':[{k:r[k] for k in ('cumulative_fresh_targets','wall_seconds','selected_step','static_test','gate_enabled','confidence_test','contexts')} for r in summary['runs']]},ensure_ascii=False,indent=2))
