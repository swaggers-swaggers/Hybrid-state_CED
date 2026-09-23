"""Nine-category KL: teacher Top8 IDs and the complete remaining vocabulary."""
import torch

def teacher_probs(top_logits,normalizers,temperature=1.):
    col={1.:0,2.:1}[float(temperature)]
    p=(top_logits.detach().float()/temperature-normalizers.detach().float()[...,col,None]).exp()
    total=p.sum(-1,keepdim=True)
    if not torch.isfinite(p).all() or bool((total>1.00002).any()): raise ValueError('Invalid sparse teacher mass')
    p=torch.cat([p,(1-total).clamp_min(0)],-1)
    return p/p.sum(-1,keepdim=True)

def student_log_probs(logits,ids,temperature=1.):
    z=logits.float()/temperature
    if ids.shape[-1]!=8 or ids.shape[:-1]!=z.shape[:-1]: raise ValueError('Top8 shape mismatch')
    selected=z.gather(-1,ids)
    rest=z.scatter(-1,ids,float('-inf')).logsumexp(-1,keepdim=True)
    # logsumexp(rest) avoids subtracting nearly equal probabilities and keeps tail gradients.
    return torch.cat([selected,rest],-1).log_softmax(-1)

def sparse_kl(logits,ids,teacher_logits,normalizers,temperature=1.):
    p=teacher_probs(teacher_logits,normalizers,temperature)
    q=student_log_probs(logits,ids,temperature)
    return (torch.special.xlogy(p,p)-p*q).sum(-1)*temperature**2

@torch.no_grad()
def proximity(logits,ids,teacher_logits,normalizers,minimum_overlap=6,maximum_kl=.05):
    predicted=logits.topk(8,dim=-1).indices
    overlap=(predicted[..., :,None]==ids[...,None,:]).any(-1).sum(-1)
    kl=sparse_kl(logits,ids,teacher_logits,normalizers,1.)
    return (overlap>=minimum_overlap)&(kl<=maximum_kl),overlap,kl
