"""Lightweight plan, explicit experiment semantics and checkpoint identity."""
import json
import math
from pathlib import Path
from ced_training.protocol import sha256
SEMANTICS='top8_tail_teacher_forced_qualified_exit_immediate_full_v1'

def validate(c):
    if c['temperature'] not in (1.,2.): raise ValueError('Only T=1/2 normalizers were stored')
    if type(c['minimum_overlap']) is not int or not 1<=c['minimum_overlap']<=8: raise ValueError('Invalid overlap')
    for k in ('maximum_top8_kl','readout_lr','projection_lr','gate_lr','gradient_clip'):
        if not math.isfinite(c[k]) or c[k]<=0: raise ValueError(k)
    for k in ('accumulation_positions','gate_epochs','gate_batch_size','minimum_accepted','eval_prompts'):
        if type(c[k]) is not int or c[k]<1: raise ValueError(k)
    if not 0<c['risk_limit']<1 or not c['threshold_grid']: raise ValueError('Invalid calibration')
    if any(not 0<=x<=1 for x in c['threshold_grid']): raise ValueError('Invalid threshold')
    if any(type(x) is not int or x<1 for x in c['decode_lengths']): raise ValueError('Invalid decode length')
    if c['maximum_artifact_bytes']>2_000_000_000 or c['minimum_free_bytes']<20_000_000_000: raise ValueError('Disk safeguards cannot be relaxed')

def plan(root,c,start,count):
    validate(c)
    path=Path(root)/c['data_path']/'manifest.json'
    if sha256(path)!=c['manifest_sha256']: raise ValueError('Dataset identity changed')
    m=json.loads(path.read_text());n=m['splits']['train']['effective_targets']
    if start<0 or count<1 or start+count>n: raise ValueError('Training interval outside dataset')
    return {'status':'PLAN_ONLY_NO_TRAINING','semantics':SEMANTICS,'interval':[start,start+count],
            'teacher_forcing':True,'kv_trigger':{'minimum_top8_overlap':c['minimum_overlap'],'maximum_top8_plus_tail_kl_T1':c['maximum_top8_kl']},
            'kv_loss':'Only next-step full-model output Top8+tail KL; no KV MSE or GDN alignment',
            'pair':'Qualified exit at j -> input teacher completion[j] -> full output supervises teacher logits[j+1]',
            'truncate_graph':'After each completed pair; preserve cache values, never teacher-repair the cache',
            'stages':['modules','confidence','calibrate','evaluate'],'config':c}

def pair_action(pending,qualified,has_successor):
    if pending:return 'full_pair'
    if qualified and has_successor:return 'exit'
    return 'full'
