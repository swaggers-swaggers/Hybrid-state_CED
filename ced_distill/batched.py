"""Shared-weight response batches and checkpointed, multi-step KV credit assignment.

A qualified anchor projects once, followed by H full steps. Each successor loss
backpropagates to that anchor, including across recomputation boundaries. No new
anchor is inserted inside its rollout. Teacher forcing and cache values persist.
"""
import copy
import itertools
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .losses import kl_from_probs,proximity
from .training import tensors,prefix_tokens,initial_cache

CACHE_FIELDS=('keys','values','conv_states','recurrent_states')


def cache_copy(cache):
    result=copy.copy(cache)
    result.layers=[copy.copy(layer) for layer in cache.layers]
    return result


def merge_prefixes(caches,lengths):
    """Left-pad attention KV only; recurrent states have no padding/time axis."""
    result=cache_copy(caches[0]);maximum=max(lengths)
    for index,layer in enumerate(result.layers):
        for name in CACHE_FIELDS:
            values=[getattr(cache.layers[index],name,None) for cache in caches]
            if values[0] is None:continue
            if name in ('keys','values'):
                values=[F.pad(value,(0,0,maximum-length,0)) for value,length in zip(values,lengths)]
            setattr(layer,name,torch.cat(values,0))
        if hasattr(layer,'max_batch_size'):layer.max_batch_size=len(caches)
    return result


def cache_spec(cache):
    return [(i,name) for i in range(12,len(cache.layers)) for name in CACHE_FIELDS
            if getattr(cache.layers[i],name,None) is not None]


def flatten_cache(cache,spec):return tuple(getattr(cache.layers[i],name) for i,name in spec)


def restore_cache(template,spec,values):
    cache=cache_copy(template)
    for (i,name),value in zip(spec,values):setattr(cache.layers[i],name,value)
    return cache


def lane_where(mask,yes,no):
    return torch.where(mask.reshape(-1,*([1]*(yes.ndim-1))),yes,no)


def schedule_rollouts(labels,lengths,horizon):
    """CPU control: an anchor and exactly min(H, remaining) full successors."""
    batch=len(lengths);steps=len(labels);remaining=[0]*batch
    exits=[];supervision=[];detach=[]
    for step in range(steps):
        e=[];s=[];d=[]
        for lane,length in enumerate(lengths):
            valid=step<length
            supervise=valid and remaining[lane]>0
            leave=valid and not supervise and labels[step][lane] and step+1<length
            if supervise:remaining[lane]-=1
            elif leave:remaining[lane]=min(horizon,length-step-1)
            e.append(leave);s.append(supervise);d.append(remaining[lane]==0)
        exits.append(e);supervision.append(s);detach.append(d)
    if any(remaining):raise AssertionError('Unfinished projection rollout')
    return exits,supervision,detach


class ResponseBatch:
    def __init__(self,rt,rows):
        self.rows=rows;self.targets=[tensors(row,rt.config['temperature']) for row in rows]
        self.lengths=[row['end']-row['start'] for row in rows];self.steps=max(self.lengths)
        self.prefix_lengths=[prefix_tokens(t,row).shape[1] for t,row in zip(self.targets,rows)]
        prefixes=[initial_cache(rt,t,row) for t,row in zip(self.targets,rows)]
        self.template=merge_prefixes(prefixes,self.prefix_lengths);del prefixes
        self.prefix_width=max(self.prefix_lengths);device=self.targets[0]['tokens'].device
        self.valid=torch.arange(self.steps,device=device)[None,:]<torch.tensor(self.lengths,device=device)[:,None]
        prefix_valid=torch.arange(self.prefix_width,device=device)[None,:]>=torch.tensor(
            [self.prefix_width-n for n in self.prefix_lengths],device=device)[:,None]
        self.key_valid=torch.cat((prefix_valid,self.valid),1)
        self.positions=torch.tensor(self.prefix_lengths,device=device)[:,None]+torch.arange(self.steps,device=device)[None,:]
        self.inputs=torch.stack([F.pad(t['tokens'][row['start']-1:row['end']-1],(0,self.steps-n))
                                for t,row,n in zip(self.targets,rows,self.lengths)])
        self.ids=torch.stack([F.pad(t['ids'][row['start']:row['end']],(0,0,0,self.steps-n))
                             for t,row,n in zip(self.targets,rows,self.lengths)])
        self.probs=torch.stack([F.pad(t['probs'][row['start']:row['end']],(0,0,0,self.steps-n))
                               for t,row,n in zip(self.targets,rows,self.lengths)])
        self.spec=cache_spec(self.template)

    def context(self,rt,hidden,cache,start,end):
        pos=self.positions[:,start:end]
        rope=rt.lm.rotary_emb(hidden,pos[None,:,:].expand(3,-1,-1))
        width=self.prefix_width+end
        columns=torch.arange(width,device=hidden.device)
        queries=self.prefix_width+torch.arange(start,end,device=hidden.device)
        causal=self.key_valid[:,:width,None].transpose(1,2) & (columns[None,:]<=queries[:,None])[None,:,:]
        return dict(cache=cache,pos=pos,rope=rope,causal=causal[:,None,:,:],linear=None)

    @torch.no_grad()
    def shallow(self,rt):
        cache=cache_copy(self.template)
        # Cached GDN convolution kernels mutate their input tensors.
        for layer in cache.layers[:12]:
            if getattr(layer,'conv_states',None) is not None:layer.conv_states=layer.conv_states.clone()
        chunks=[]
        for start in range(0,self.steps,rt.config['readout_chunk_size']):
            end=min(self.steps,start+rt.config['readout_chunk_size'])
            with torch.autocast('cuda',dtype=torch.bfloat16):
                h=rt.lm.embed_tokens(self.inputs[:,start:end]);state=self.context(rt,h,cache,start,end)
                for index in range(12):h=rt.layer(index,h,state)
            chunks.append(h)
        return torch.cat(chunks,1),len(chunks)


def deep_step(rt,batch,cache,hidden,step,exits,supervision,detach,need_logits):
    """Functional boundary: never mutate any cache tensor owned by the caller."""
    full=cache_copy(cache)
    for layer in full.layers[12:]:
        if getattr(layer,'conv_states',None) is not None:layer.conv_states=layer.conv_states.clone()
    state=batch.context(rt,hidden,full,step,step+1)
    logits=rt.finish(hidden,state,compute_logits=need_logits)
    loss=hidden.new_zeros((),dtype=torch.float32)
    if need_logits:
        # Padding/anchor rows never contribute, including to the denominator.
        loss=(kl_from_probs(logits,batch.ids[:,step],batch.probs[:,step],
                            rt.config['temperature'])*supervision).sum()
    if exits is not None:
        projected=cache_copy(cache)
        rt.project(hidden,{**state,'cache':projected})
    held=(~batch.valid[:,step]) if exits is None else (exits | ~batch.valid[:,step])
    for index,name in batch.spec:
        new=getattr(full.layers[index],name)
        if name in ('keys','values'):
            if exits is not None:new=lane_where(exits,getattr(projected.layers[index],name),new)
        else:new=lane_where(held,getattr(cache.layers[index],name),new)
        # Only completed horizons detach, never activation-checkpoint boundaries.
        setattr(full.layers[index],name,lane_where(detach,new.detach(),new))
    return full,loss


def projection_objective(rt,batch,hidden,labels,train=True,use_checkpoint=True,loss_steps=None):
    exits,supervision,detach=schedule_rollouts(labels,batch.lengths,rt.config['projection_horizon'])
    device=hidden.device
    es=torch.tensor(exits,device=device);ss=torch.tensor(supervision,device=device);ds=torch.tensor(detach,device=device)
    values=flatten_cache(batch.template,batch.spec);total=hidden.new_zeros((),dtype=torch.float32)
    segment=rt.config['checkpoint_steps']
    for begin in range(0,batch.steps,segment):
        end=min(batch.steps,begin+segment)
        # Bind segment endpoints: backward recomputation happens after the loop.
        def run(*inputs,begin=begin,end=end):
            cache=restore_cache(batch.template,batch.spec,inputs);loss=hidden.new_zeros((),dtype=torch.float32)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                for step in range(begin,end):
                    need=any(supervision[step]) and (loss_steps is None or step in loss_steps)
                    cache,current=deep_step(rt,batch,cache,hidden[:,step:step+1],step,
                        es[step] if any(exits[step]) else None,ss[step],ds[step],need)
                    loss=loss+current
            return (loss,*flatten_cache(cache,batch.spec))
        result=checkpoint(run,*values,use_reentrant=False,preserve_rng_state=False) if train and use_checkpoint else run(*values)
        total=total+result[0];values=result[1:]
    return total,{'kv_projections':sum(map(sum,exits)),'kv_supervised_positions':sum(map(sum,supervision)),
                  'projection_horizon':rt.config['projection_horizon']}


def module_pass_batched(rt,data,split,start,count,output=None,train=False,max_responses=None,storage_guard=None,budget=None):
    c=rt.config;r=rt.runner;r.requires_grad_(False)
    if train:r.readout_map.requires_grad_(True);r.kv_projectors.requires_grad_(True)
    groups=[list(r.readout_map.parameters()),list(r.kv_projectors.parameters())]
    optimizers=[torch.optim.AdamW(p,lr=lr,weight_decay=c['weight_decay'],fused=True)
                for p,lr in zip(groups,(c['readout_lr'],c['projection_lr']))] if train else []
    stats=dict(positions=0,qualified_candidates=0,kv_projections=0,kv_supervised_positions=0,
               readout_updates=0,projection_updates=0,readout_batches=0,shallow_batches=0,response_batches=0,
               maximum_response_batch=0,projection_horizon=c['projection_horizon'],checkpoint_steps=c['checkpoint_steps'])
    sums=torch.zeros(4,device='cuda',dtype=torch.float64)
    rows=iter(data.rows(split,start,count,max_responses))
    while True:
        if budget is not None and budget.boundary(stats['positions']):break
        group=list(itertools.islice(rows,c['response_batch_size']))
        if not group:break
        # Preserve an exact contiguous consumed interval; stop only between response batches.
        remaining=(budget.target if budget is not None else count)-stats['positions'];trimmed=[]
        for row in group:
            if remaining<=0:break
            n=min(remaining,row['end']-row['start']);row={**row,'end':row['start']+n}
            trimmed.append(row);remaining-=n
        batch=ResponseBatch(rt,trimmed);hidden,shallow_count=batch.shallow(rt)
        n=sum(batch.lengths);labels=torch.zeros_like(batch.valid)
        stats['shallow_batches']+=shallow_count;stats['response_batches']+=1
        stats['maximum_response_batch']=max(stats['maximum_response_batch'],len(trimmed))
        for begin in range(0,batch.steps,c['readout_chunk_size']):
            end=min(batch.steps,begin+c['readout_chunk_size']);valid=batch.valid[:,begin:end]
            with torch.set_grad_enabled(train),torch.autocast('cuda',dtype=torch.bfloat16):
                # Keep the vocabulary projection bounded; padding is removed first.
                z=rt.readout_all(hidden[:,begin:end][valid])
                losses=kl_from_probs(z,batch.ids[:,begin:end][valid],batch.probs[:,begin:end][valid],c['temperature'])
                good,overlap,top1=proximity(z.detach(),batch.ids[:,begin:end][valid],c['minimum_overlap'])
                labels[:,begin:end][valid]=good
                sums[0]+=losses.detach().double().sum();sums[2]+=overlap.double().sum();sums[3]+=top1.double().sum()
                if train:losses.sum().backward()
            stats['readout_batches']+=1
            del z,losses,good,overlap,top1
        decisions=labels.transpose(0,1).tolist();stats['qualified_candidates']+=sum(map(sum,decisions))
        with torch.set_grad_enabled(train):
            loss,deep_stats=projection_objective(rt,batch,hidden,decisions,train=train)
            sums[1]+=loss.detach().double()
            if train and deep_stats['kv_supervised_positions']:loss.backward()
        for key in ('kv_projections','kv_supervised_positions'):stats[key]+=deep_stats[key]
        if train:
            for index,(optimizer,params,denominator) in enumerate(zip(optimizers,groups,(n,deep_stats['kv_supervised_positions']))):
                if denominator:
                    for param in params:
                        if param.grad is not None:param.grad.div_(denominator)
                    torch.nn.utils.clip_grad_norm_(params,c['gradient_clip'],error_if_nonfinite=True)
                    optimizer.step();stats['readout_updates' if index==0 else 'projection_updates']+=1
                optimizer.zero_grad(set_to_none=True)
        stats['positions']+=n
        if output:
            if storage_guard:storage_guard(4096)
            with (Path(output)/'train.jsonl').open('a') as f:f.write(json.dumps(stats,allow_nan=False)+'\n')
        del batch,hidden,loss
    values=sums.tolist();n=max(stats['positions'],1)
    if not all(torch.isfinite(torch.tensor(values))):raise FloatingPointError('Non-finite training metrics')
    stats.update(readout_kl_sum=values[0],rollout_kl_sum=values[1],readout_kl=values[0]/n,
                 rollout_full_kl=values[1]/stats['kv_supervised_positions'] if stats['kv_supervised_positions'] else None,
                 mean_top8_overlap=values[2]/n,top1_agreement=values[3]/n,
                 qualification_rate=stats['qualified_candidates']/n,
                 status='FINISHED' if stats['kv_supervised_positions'] else 'NO_QUALIFIED_KV_ROLLOUTS')
    if train and budget is None and max_responses is None and stats['positions']!=count:raise AssertionError('Effective token budget mismatch')
    if budget is not None:
        budget.boundary(stats['positions']);stats['budget']=budget.report()
    stats['actual_interval']=[start,start+stats['positions']];r.requires_grad_(False)
    return stats
