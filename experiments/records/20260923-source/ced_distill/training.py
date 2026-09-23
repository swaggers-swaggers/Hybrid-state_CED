"""Teacher-forced conditional pairs; no KV labels, GDN controls, or hidden repair."""
import json
import math
from pathlib import Path
import numpy as np
import torch
from .losses import sparse_kl,proximity
from .runtime import detach_cache
from .protocol import pair_action

def tensors(row):
    result={key:torch.as_tensor(row[key],device='cuda',dtype=torch.long if key in ('prompt','tokens','ids') else torch.float32)
            for key in ('prompt','tokens','ids','logits','normalizers')}
    return result

def initial_cache(runtime,t,row):
    prefix=torch.cat([t['prompt'],t['tokens'][:row['start']-1]])[None,:]
    cache,_=runtime.prefill(prefix)
    return cache

def kd(logits,t,j,temperature):
    return sparse_kl(logits,t['ids'][j:j+1],t['logits'][j:j+1],t['normalizers'][j:j+1],temperature).mean()

def qualify(logits,t,j,c):
    return proximity(logits,t['ids'][j:j+1],t['logits'][j:j+1],t['normalizers'][j:j+1],c['minimum_overlap'],c['maximum_top8_kl'])

def module_pass(rt,data,split,start,count,output=None,train=False,max_responses=None,storage_guard=None,budget=None):
    c=rt.config;r=rt.runner
    r.requires_grad_(False)
    if train:
        r.readout_map.requires_grad_(True);r.kv_projectors.requires_grad_(True)
    groups=[list(r.readout_map.parameters()),list(r.kv_projectors.parameters())]
    optimizers=[torch.optim.AdamW(p,lr=lr,weight_decay=c['weight_decay']) for p,lr in zip(groups,(c['readout_lr'],c['projection_lr']))] if train else []
    stats={'positions':0,'readout_kl_sum':0.,'full_pair_kl_sum':0.,'qualified_candidates':0,'kv_pairs':0,
           'readout_updates':0,'projection_updates':0,'last_position_candidates_without_successor':0}
    accumulated=[0,0];update_positions=0;stopped=False
    def flush():
        nonlocal update_positions
        if not train:return
        for index,(optimizer,params) in enumerate(zip(optimizers,groups)):
            if accumulated[index]:
                for p in params:
                    if p.grad is not None:p.grad.div_(accumulated[index])
                torch.nn.utils.clip_grad_norm_(params,c['gradient_clip'],error_if_nonfinite=True)
                optimizer.step();stats['readout_updates' if index==0 else 'projection_updates']+=1
            optimizer.zero_grad(set_to_none=True);accumulated[index]=0
        if output:
            if storage_guard:storage_guard(4096)
            with (Path(output)/'train.jsonl').open('a') as f:f.write(json.dumps(stats,allow_nan=False)+'\n')
        update_positions=0
    for row in data.rows(split,start,count,max_responses):
        t=tensors(row);cache=initial_cache(rt,t,row);pending=False
        for j in range(row['start'],row['end']):
            if budget is not None and not pending and budget.boundary(stats['positions']):
                stopped=True;break
            with torch.set_grad_enabled(train),torch.autocast('cuda',dtype=torch.bfloat16):
                h,state=rt.begin(t['tokens'][j-1:j][None,:],cache)
                logits=rt.readout(h)
                loss=kd(logits,t,j,c['temperature'])
                good,overlap,distance=qualify(logits,t,j,c)
                accepted=bool(good.item());stats['qualified_candidates']+=int(accepted)
                stats['readout_kl_sum']+=float(loss.detach());stats['positions']+=1;update_positions+=1
                if train:loss.backward();accumulated[0]+=1
                has_successor=j+1<row['end'] and (budget is None or stats['positions']<budget.target)
                action=pair_action(pending,accepted,has_successor)
                if action=='full_pair':
                    # Even if this step's readout also qualifies, it MUST run every deep block.
                    full_logits=rt.finish(h,state)
                    pair_loss=kd(full_logits,t,j,c['temperature'])
                    stats['full_pair_kl_sum']+=float(pair_loss.detach());stats['kv_pairs']+=1
                    if train:pair_loss.backward();accumulated[1]+=1
                    pending=False
                elif action=='exit':
                    rt.project(h.detach(),state);pending=True
                else:
                    if accepted:stats['last_position_candidates_without_successor']+=1
                    # No graph through unqualified full steps. Their cache values are still retained.
                    with torch.no_grad():rt.finish(h,state)
            if not pending:
                detach_cache(cache)
                if update_positions>=c['accumulation_positions']:flush()
        if pending:raise AssertionError('Unsupervised projected KV at sequence/budget boundary')
        del cache
        if stopped:break
    flush()
    stats['readout_kl']=stats['readout_kl_sum']/max(stats['positions'],1)
    stats['next_full_kl']=stats['full_pair_kl_sum']/stats['kv_pairs'] if stats['kv_pairs'] else None
    stats['status']='FINISHED' if stats['kv_pairs'] else 'NO_QUALIFIED_KV_PAIRS'
    if train and budget is None and stats['positions']!=count:raise AssertionError('Effective token budget mismatch')
    if budget is not None:
        budget.boundary(stats['positions']);stats['budget']=budget.report()
        stats['actual_interval']=[start,start+stats['positions']]
    r.requires_grad_(False)
    return stats

@torch.no_grad()
def gate_features(rt,data,split,max_responses=None):
    # h12 is independent of deep-cache history; both paths fully execute layers 1..12.
    features=[];labels=[]
    for row in data.rows(split,max_responses=max_responses):
        t=tensors(row);cache=initial_cache(rt,t,row)
        for j in range(row['start'],row['end']):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                h,state=rt.begin(t['tokens'][j-1:j][None,:],cache)
                logits=rt.readout(h);good,_,_=qualify(logits,t,j,rt.config)
                # Need only shallow cache for features: deeper layers are not subsequently read here.
            features.append(h.reshape(-1).float().cpu());labels.append(float(good.item()))
        del cache
    if not features:raise ValueError('Empty gate features')
    return torch.stack(features),torch.tensor(labels)

def confidence_metrics(head,x,y,batch_size):
    scores=[]
    with torch.no_grad():
        for i in range(0,len(x),batch_size):scores.append(head(x[i:i+batch_size].cuda()).reshape(-1).cpu())
    logits=torch.cat(scores);p=logits.sigmoid()
    return {'bce':torch.nn.functional.binary_cross_entropy_with_logits(logits,y).item(),'brier':((p-y)**2).mean().item(),
            'positive_rate':y.mean().item(),'positions':len(y)},p

def train_confidence(rt,data,max_responses=None):
    c=rt.config;head=rt.runner.confidence_head
    rt.runner.requires_grad_(False)
    x,y=gate_features(rt,data,'gate',max_responses);dx,dy=gate_features(rt,data,'dev',max_responses)
    head.requires_grad_(True);optimizer=torch.optim.AdamW(head.parameters(),lr=c['gate_lr'])
    initial,_=confidence_metrics(head,dx,dy,c['gate_batch_size'])
    best=initial['bce'];chosen={k:v.detach().clone() for k,v in head.state_dict().items()};history=[]
    for epoch in range(c['gate_epochs']):
        order=torch.randperm(len(y),generator=torch.Generator().manual_seed(c['seed']+epoch))
        for start in range(0,len(y),c['gate_batch_size']):
            idx=order[start:start+c['gate_batch_size']];optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(head(x[idx].cuda()).reshape(-1),y[idx].cuda())
            loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),c['gradient_clip'],error_if_nonfinite=True);optimizer.step()
        metrics,_=confidence_metrics(head,dx,dy,c['gate_batch_size']);history.append({'epoch':epoch+1,**metrics})
        if metrics['bce']<best:
            best=metrics['bce'];chosen={k:v.detach().clone() for k,v in head.state_dict().items()}
    head.load_state_dict(chosen);head.requires_grad_(False)
    return {'status':'CONFIDENCE_TRAINED','initial_dev':initial,'warm_started':True,'history':history,'training_positions':len(y),'label':'Top8 overlap AND Top8+tail KL; not Top1 match'}
