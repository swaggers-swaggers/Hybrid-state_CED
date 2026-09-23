"""All-position detector supervision and warm-started confidence fitting."""
import torch
from .losses import teacher_probs,kl_from_probs,proximity


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
    # Import lazily: batched training reuses the sparse-target conversion above.
    from .batched import module_pass_batched
    return module_pass_batched(rt,data,split,start,count,output,train,max_responses,storage_guard,budget)


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
