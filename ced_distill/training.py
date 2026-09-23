"""All-position detector distillation, chunked shallow pass, ordered deep KV pairs."""
import copy
import json
from pathlib import Path
import torch
from .losses import teacher_probs,kl_from_probs,proximity
from .runtime import detach_cache
from .protocol import pair_action


def tensors(row,temperature=1.):
    result={key:torch.as_tensor(row[key],device='cuda',dtype=torch.long if key in ('prompt','tokens','ids') else torch.float32)
            for key in ('prompt','tokens','ids','logits','normalizers')}
    # Validate the sparse targets once per response, not on every GPU token step.
    result['probs']=teacher_probs(result['logits'],result['normalizers'],temperature)
    result['temperature']=temperature
    return result


def prefix_tokens(t,row):
    return torch.cat([t['prompt'],t['tokens'][:row['start']-1]])[None,:]


def initial_cache(runtime,t,row):
    cache,_=runtime.prefill(prefix_tokens(t,row))
    return cache


def kd(logits,t,j,temperature):
    if t['temperature']!=temperature:raise ValueError('Target temperature differs')
    return kl_from_probs(logits,t['ids'][j:j+1],t['probs'][j:j+1],temperature).mean()


def qualify(logits,t,j,c):
    return proximity(logits,t['ids'][j:j+1],c['minimum_overlap'])


@torch.no_grad()
def hidden_chunks(rt,data,split,start=0,count=None,max_responses=None):
    """No deep execution is needed to observe h12 under the teacher prefix."""
    for row in data.rows(split,start,count,max_responses):
        t=tensors(row);cache=rt.cache()
        with torch.autocast('cuda',dtype=torch.bfloat16):rt.begin(prefix_tokens(t,row),cache)
        for j in range(row['start'],row['end'],rt.config['readout_chunk_size']):
            end=min(row['end'],j+rt.config['readout_chunk_size'])
            with torch.autocast('cuda',dtype=torch.bfloat16):
                hidden,_=rt.begin(t['tokens'][j-1:end-1][None,:],cache)
            yield row,t,j,end,hidden
        del cache


def module_pass(rt,data,split,start,count,output=None,train=False,max_responses=None,storage_guard=None,budget=None):
    c=rt.config;r=rt.runner;r.requires_grad_(False)
    if train:
        r.readout_map.requires_grad_(True);r.kv_projectors.requires_grad_(True)
    groups=[list(r.readout_map.parameters()),list(r.kv_projectors.parameters())]
    optimizers=[torch.optim.AdamW(p,lr=lr,weight_decay=c['weight_decay'],fused=True)
                for p,lr in zip(groups,(c['readout_lr'],c['projection_lr']))] if train else []
    stats={'positions':0,'qualified_candidates':0,'kv_pairs':0,'readout_updates':0,'projection_updates':0,
           'last_position_candidates_without_successor':0,'readout_batches':0,'shallow_batches':0}
    # Sums stay on the GPU. Only the chunk's branch labels cross to CPU.
    sums=torch.zeros(4,device='cuda',dtype=torch.float64)
    accumulated=[0,0];update_positions=0;stopped=False
    def flush():
        nonlocal update_positions
        if not train:return
        for index,(optimizer,params) in enumerate(zip(optimizers,groups)):
            if accumulated[index]:
                for param in params:
                    if param.grad is not None:param.grad.div_(accumulated[index])
                torch.nn.utils.clip_grad_norm_(params,c['gradient_clip'],error_if_nonfinite=True)
                optimizer.step();stats['readout_updates' if index==0 else 'projection_updates']+=1
            optimizer.zero_grad(set_to_none=True);accumulated[index]=0
        if output:
            if storage_guard:storage_guard(4096)
            with (Path(output)/'train.jsonl').open('a') as f:f.write(json.dumps(stats,allow_nan=False)+'\n')
        update_positions=0
    for row in data.rows(split,start,count,max_responses):
        if budget is not None and budget.boundary(stats['positions']):stopped=True;break
        t=tensors(row,c['temperature']);deep_cache=initial_cache(rt,t,row)
        # Both copies hold exactly the same prefix. Only the first 12 layers of
        # shallow_cache advance in chunks; deep_cache advances one position at a time.
        shallow_cache=copy.deepcopy(deep_cache);pending=False;j=row['start']
        while j<row['end']:
            stop_now=budget is not None and budget.boundary(stats['positions'])
            if stop_now and not pending:stopped=True;break
            size=min(c['readout_chunk_size'],row['end']-j)
            if budget is not None:size=min(size,max(1,budget.target-stats['positions']))
            if stop_now:size=1  # Finish a pending pair without supervising an unused batch.
            end=j+size
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                hidden,state=rt.begin(t['tokens'][j-1:end-1][None,:],shallow_cache)
            stats['shallow_batches']+=1
            with torch.set_grad_enabled(train),torch.autocast('cuda',dtype=torch.bfloat16):
                logits=rt.readout_all(hidden)
                losses=kl_from_probs(logits,t['ids'][j:end],t['probs'][j:end],c['temperature'])
                good,overlap,top1=proximity(logits.detach(),t['ids'][j:end],c['minimum_overlap'])
                sums[0]+=losses.detach().double().sum();sums[2]+=overlap.double().sum();sums[3]+=top1.double().sum()
                if train:losses.sum().backward();accumulated[0]+=size
            labels=good.tolist();del logits,losses,good,overlap,top1
            stats['readout_batches']+=1;stats['qualified_candidates']+=sum(labels)
            for offset,accepted in enumerate(labels):
                pos=j+offset;h=hidden[:,offset:offset+1];ctx=rt.deep_context(state,deep_cache,offset)
                stats['positions']+=1;update_positions+=1
                successor=pos+1<row['end'] and (budget is None or stats['positions']<budget.target)
                action=pair_action(pending,accepted,successor)
                with torch.set_grad_enabled(train),torch.autocast('cuda',dtype=torch.bfloat16):
                    if action=='full_pair':
                        full=rt.finish(h,ctx);loss=kd(full,t,pos,c['temperature'])
                        sums[1]+=loss.detach().double();stats['kv_pairs']+=1
                        if train:loss.backward();accumulated[1]+=1
                        pending=False;del full,loss
                    elif action=='exit':
                        rt.project(h,ctx);pending=True
                    else:
                        if accepted:stats['last_position_candidates_without_successor']+=1
                        with torch.no_grad():rt.finish(h,ctx,compute_logits=False)
                if not pending:detach_cache(deep_cache)
            # Neither projector parameters nor an outstanding graph are changed
            # between the exit and its immediate full successor (including chunk edges).
            if not pending and update_positions>=c['accumulation_positions']:flush()
            j=end
        if pending:raise AssertionError('Unsupervised projected KV at sequence/budget boundary')
        del deep_cache,shallow_cache
        if stopped:break
    flush();values=sums.tolist();n=max(stats['positions'],1)
    if not all(torch.isfinite(torch.tensor(values))):raise FloatingPointError('Non-finite training metrics')
    stats.update(readout_kl_sum=values[0],full_pair_kl_sum=values[1],readout_kl=values[0]/n,
                 next_full_kl=values[1]/stats['kv_pairs'] if stats['kv_pairs'] else None,
                 mean_top8_overlap=values[2]/n,top1_agreement=values[3]/n,
                 qualification_rate=stats['qualified_candidates']/n,
                 status='FINISHED' if stats['kv_pairs'] else 'NO_QUALIFIED_KV_PAIRS')
    if train and budget is None and stats['positions']!=count:raise AssertionError('Effective token budget mismatch')
    if budget is not None:
        budget.boundary(stats['positions']);stats['budget']=budget.report()
    stats['actual_interval']=[start,start+stats['positions']]
    r.requires_grad_(False)
    return stats


@torch.no_grad()
def gate_features(rt,data,split,max_responses=None):
    features=[];labels=[]
    for row,t,j,end,h in hidden_chunks(rt,data,split,max_responses=max_responses):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=rt.readout_all(h);good,_,_=proximity(logits,t['ids'][j:end],rt.config['minimum_overlap'])
        features.append(h.reshape(-1,h.shape[-1]).float().cpu());labels.append(good.float().cpu())
    if not features:raise ValueError('Empty gate features')
    return torch.cat(features),torch.cat(labels)


def confidence_metrics(head,x,y,batch_size):
    scores=[]
    with torch.no_grad():
        for i in range(0,len(x),batch_size):scores.append(head(x[i:i+batch_size].cuda()).reshape(-1).cpu())
    logits=torch.cat(scores);p=logits.sigmoid()
    return {'bce':torch.nn.functional.binary_cross_entropy_with_logits(logits,y).item(),'brier':((p-y)**2).mean().item(),
            'positive_rate':y.mean().item(),'positions':len(y)},p


def train_confidence(rt,data,max_responses=None):
    c=rt.config;head=rt.runner.confidence_head;rt.runner.requires_grad_(False)
    x,y=gate_features(rt,data,'gate',max_responses);dx,dy=gate_features(rt,data,'dev',max_responses)
    head.requires_grad_(True)
    optimizer=torch.optim.AdamW(head.parameters(),lr=c['gate_lr'],fused=next(head.parameters()).is_cuda)
    initial,_=confidence_metrics(head,dx,dy,c['gate_batch_size'])
    best=initial['bce'];chosen={k:v.detach().clone() for k,v in head.state_dict().items()};history=[]
    for epoch in range(c['gate_epochs']):
        order=torch.randperm(len(y),generator=torch.Generator().manual_seed(c['seed']+epoch))
        for start in range(0,len(y),c['gate_batch_size']):
            idx=order[start:start+c['gate_batch_size']];optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(head(x[idx].cuda()).reshape(-1),y[idx].cuda())
            loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),c['gradient_clip'],error_if_nonfinite=True);optimizer.step()
        metrics,_=confidence_metrics(head,dx,dy,c['gate_batch_size']);history.append({'epoch':epoch+1,**metrics})
        if metrics['bce']<best:best=metrics['bce'];chosen={k:v.detach().clone() for k,v in head.state_dict().items()}
    head.load_state_dict(chosen);head.requires_grad_(False)
    return {'status':'CONFIDENCE_TRAINED','initial_dev':initial,'warm_started':True,'history':history,
            'training_positions':len(y),'label':'Top8 overlap >=6 OR Top1 match'}
