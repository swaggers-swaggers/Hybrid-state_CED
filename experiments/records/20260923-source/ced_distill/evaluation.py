"""Actual adaptive outputs and paired generation cost; no GDN/KV oracle controls."""
import math
import time
import statistics
import torch
from .training import tensors,initial_cache,kd,gate_features,confidence_metrics
from .losses import proximity
from ced_training.protocol import calibrate

@torch.no_grad()
def calibrate_gate(rt,data,max_responses=None):
    x,y=gate_features(rt,data,'calibration',max_responses)
    metrics,scores=confidence_metrics(rt.runner.confidence_head,x,y,rt.config['gate_batch_size'])
    c=rt.config;result=calibrate(scores.tolist(),y.tolist(),c['threshold_grid'],c['risk_limit'],c['minimum_accepted'])
    result['interpretation']='Empirical Top8+tail qualification risk, not Top1 disagreement or a formal statistical bound.'
    for row in result['curve']:row['top8_violation_rate']=row.pop('disagreement')
    result['metrics']=metrics;rt.threshold=result['threshold']
    return result

@torch.no_grad()
def static_adaptive(rt,data,max_responses=None):
    n=exits=0;kl=nll=teacher_nll=0.
    for row in data.rows('test',max_responses=max_responses):
        t=tensors(row);cache=initial_cache(rt,t,row)
        for j in range(row['start'],row['end']):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits,exited=rt.adaptive(t['tokens'][j-1:j][None,:],cache)
                kl+=float(kd(logits,t,j,1.))
            nll+=float(torch.nn.functional.cross_entropy(logits.float(),t['tokens'][j:j+1]))
            teacher_nll+=float(row['normalizers'][j,0]-row['sampled_logit'][j])
            n+=1;exits+=exited
    return {'positions':n,'exits':exits,'exit_rate':exits/max(n,1),'top8_tail_kl':kl/max(n,1),
            'student_nll':nll/max(n,1),'teacher_nll':teacher_nll/max(n,1),
            'ppl_ratio':math.exp((nll-teacher_nll)/max(n,1)),
            'limitation':'Teacher-generated held-out sequences, not real-world task accuracy.'}

@torch.no_grad()
def generate(rt,prompt,length,adaptive):
    cache,initial=rt.prefill(prompt)
    token=initial.argmax(-1,keepdim=True);ids=[int(token.item())];exits=0
    # First token and prefill excluded; identical fixed decode length for comparable timing.
    torch.cuda.synchronize();begin=time.perf_counter()
    for _ in range(length):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            if adaptive:logits,exit_now=rt.adaptive(token,cache)
            else:logits=rt.runner.native(token,cache)[:,-1,:];exit_now=False
        token=logits.argmax(-1,keepdim=True);ids.append(int(token.item()));exits+=exit_now
    torch.cuda.synchronize();seconds=time.perf_counter()-begin
    return {'ids':ids,'exits':exits,'seconds':seconds,'ms_per_token':seconds*1000/length}

@torch.no_grad()
def score_own_prefix(rt,prompt,ids):
    cache,_=rt.prefill(prompt);loss=0.
    for i in range(1,len(ids)):
        token=torch.tensor([[ids[i-1]]],device='cuda')
        logits=rt.runner.native(token,cache)[:,-1,:]
        loss+=float(torch.nn.functional.cross_entropy(logits.float(),torch.tensor([ids[i]],device='cuda')))
    grams=[tuple(ids[i:i+4]) for i in range(len(ids)-3)]
    eos=rt.model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
    first_eos=next((i for i,v in enumerate(ids) if v in (eos or [])),None)
    return {'teacher_nll_on_own_prefix':loss/(len(ids)-1),'repeated_4gram_fraction':1-len(set(grams))/len(grams) if grams else 0.,
            'first_eos_index':first_eos,'note':'Teacher preference and repetition diagnostics, not external task correctness; timing continues past EOS.'}

@torch.no_grad()
def evaluate(rt,data,max_responses=None):
    rt.runner.requires_grad_(False)
    static=static_adaptive(rt,data,max_responses)
    x,y=gate_features(rt,data,'test',max_responses);gate,scores=confidence_metrics(rt.runner.confidence_head,x,y,rt.config['gate_batch_size'])
    accepted=scores>=rt.threshold if rt.threshold is not None else torch.zeros_like(y,dtype=torch.bool)
    gate.update(accepted=int(accepted.sum()),coverage=float(accepted.float().mean()),
                top8_violation_rate=float((1-y[accepted]).mean()) if accepted.any() else None)
    rows=[];seen=set()
    for row in data.rows('test',max_responses=max_responses):
        if row['prompt_id'] in seen:continue
        seen.add(row['prompt_id']);prompt=torch.tensor(row['prompt'],device='cuda',dtype=torch.long)[None,:]
        for length in rt.config['decode_lengths']:
            generate(rt,prompt,2,False);generate(rt,prompt,2,True)
            order=(False,True) if len(seen)%2 else (True,False);pair={}
            for adaptive in order:pair['adaptive' if adaptive else 'full']=generate(rt,prompt,length,adaptive)
            for name in ('full','adaptive'):pair[name]['quality']=score_own_prefix(rt,prompt,pair[name]['ids'])
            pair.update(prompt_id=row['prompt_id'],decode_tokens=length,speedup=pair['full']['seconds']/pair['adaptive']['seconds'])
            rows.append(pair)
        if len(seen)>=rt.config['eval_prompts']:break
    return {'status':'EVALUATION_FINISHED','threshold':rt.threshold,'static':static,'confidence':gate,'generations':rows,
            'median_speedup':statistics.median(r['speedup'] for r in rows),
            'quality_status':'DIAGNOSTIC_ONLY_NO_AUTOMATIC_ACCEPTANCE','state_bias_controls':False}
