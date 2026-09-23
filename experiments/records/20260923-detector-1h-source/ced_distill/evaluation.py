"""Observe every detector output; no adaptive fallback and no cost benchmark."""
import torch
from .training import hidden_chunks,gate_features,confidence_metrics
from .losses import proximity,student_log_probs
from ced_training.protocol import calibrate


@torch.no_grad()
def readout_metrics(rt,data,split='test',max_responses=None):
    # positions, KL9, NLL, teacher NLL, top1, overlap, overlap>=6, OR label,
    # tail mass absolute error, conditional Top8 KL, confidence BCE.
    sums=torch.zeros(11,device='cuda',dtype=torch.float64)
    histogram=torch.zeros(9,device='cuda',dtype=torch.long)
    for row,t,j,end,h in hidden_chunks(rt,data,split,max_responses=max_responses):
        with torch.autocast('cuda',dtype=torch.bfloat16):logits=rt.readout_all(h)
        p=t['probs'][j:end];ids=t['ids'][j:end];q=student_log_probs(logits,ids)
        kl=(torch.special.xlogy(p,p)-p*q).sum(-1)
        good,overlap,top1=proximity(logits,ids,rt.config['minimum_overlap'])
        nll=torch.nn.functional.cross_entropy(logits.float(),t['tokens'][j:end],reduction='sum')
        teacher_nll=torch.as_tensor(row['normalizers'][j:end,0]-row['sampled_logit'][j:end],device='cuda').sum()
        conditional_p=t['logits'][j:end].softmax(-1)
        conditional_q=logits.float().gather(-1,ids).log_softmax(-1)
        conditional_kl=(torch.special.xlogy(conditional_p,conditional_p)-conditional_p*conditional_q).sum(-1)
        gate_loss=torch.nn.functional.binary_cross_entropy_with_logits(rt.gate(h),good.float(),reduction='sum')
        values=[torch.tensor(end-j,device='cuda'),kl.sum(),nll,teacher_nll,top1.sum(),overlap.sum(),
                (overlap>=rt.config['minimum_overlap']).sum(),good.sum(),(p[:,-1]-q[:,-1].exp()).abs().sum(),conditional_kl.sum(),gate_loss]
        sums+=torch.stack(values).double();histogram+=torch.bincount(overlap,minlength=9)
    v=sums.tolist();n=int(v[0])
    if not n:raise ValueError('Empty detector evaluation')
    if not torch.isfinite(sums).all():raise FloatingPointError('Non-finite detector metrics')
    return {'positions':n,'top8_tail_kl':v[1]/n,'target_token_nll':v[2]/n,'teacher_target_nll':v[3]/n,
            'top1_agreement':v[4]/n,'mean_top8_overlap':v[5]/n,'top8_recall':v[5]/(8*n),
            'top8_overlap_ge6_rate':v[6]/n,'qualification_rate':v[7]/n,'tail_mass_mae':v[8]/n,
            'conditional_top8_kl':v[9]/n,'confidence_bce':v[10]/n,'top8_overlap_histogram':histogram.tolist(),
            'output_source':'block12_readout_at_every_position','temperature':1.,'confidence_filter_applied':False,
            'limitation':'Teacher-prefix held-out detector outputs; not free-generation task accuracy.'}


def metric_delta(before,after):
    if before['positions']!=after['positions']:raise ValueError('Incomparable evaluation position counts')
    keys=('top8_tail_kl','target_token_nll','top1_agreement','mean_top8_overlap','top8_recall','qualification_rate','conditional_top8_kl')
    return {k:after[k]-before[k] for k in keys}


@torch.no_grad()
def calibrate_gate(rt,data,max_responses=None):
    # Optional analysis only. Never suppress readout outputs or decide whether to evaluate.
    x,y=gate_features(rt,data,'calibration',max_responses)
    metrics,scores=confidence_metrics(rt.runner.confidence_head,x,y,rt.config['gate_batch_size'])
    c=rt.config;result=calibrate(scores.tolist(),y.tolist(),c['threshold_grid'],c['risk_limit'],c['minimum_accepted'])
    for row in result['curve']:row['criterion_violation_rate']=row.pop('disagreement')
    result.update(metrics=metrics,interpretation='Optional confidence calibration for Top8>=6 OR Top1; detector evaluation is unconditional.')
    rt.threshold=result['threshold'];return result


@torch.no_grad()
def evaluate(rt,data,max_responses=None):
    rt.runner.requires_grad_(False)
    return {'status':'DETECTOR_EVALUATION_FINISHED','detector':readout_metrics(rt,data,'test',max_responses),
            'cost_benchmark_executed':False,'state_bias_controls':False,'calibration_required':False}
