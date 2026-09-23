"""Bounded real-model engineering checks, with no optimizer or saved trained weights."""
import torch
from .training import tensors,initial_cache,kd,qualify,module_pass
from .runtime import detach_cache


def verify(rt,data):
    row=next(data.rows('train',count=8));t=tensors(row)
    if row['end']-row['start']<2:raise AssertionError('Need a complete test pair')
    j=row['start'];r=rt.runner
    # Complete split-step logits must match the original native full computation.
    with torch.no_grad():
        a=initial_cache(rt,t,row);b=initial_cache(rt,t,row)
        errors=[]
        for step in range(j,j+2):
            token=t['tokens'][step-1:step][None,:]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                h,ctx=rt.begin(token,a);manual=rt.finish(h,ctx)
                native=r.native(token,b)[:,-1,:]
            torch.testing.assert_close(manual,native,atol=.0625,rtol=0)
            errors.append(float((manual-native).abs().max()))
    del a,b
    # Synthetic forced branch tests the graph, not training eligibility; no optimizer exists here.
    r.requires_grad_(False);r.readout_map.requires_grad_(True);r.kv_projectors.requires_grad_(True)
    cache=initial_cache(rt,t,row)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        h,ctx=rt.begin(t['tokens'][j-1:j][None,:],cache)
        exit_logits=rt.readout(h);exit_loss=kd(exit_logits,t,j,rt.config['temperature'])
        actual_good,overlap,distance=qualify(exit_logits,t,j,rt.config)
        exit_loss.backward()
        readout_norm=float(r.readout_map.weight.grad.norm())
        if readout_norm<=0:raise AssertionError('Readout gradient missing')
        if any(p.grad is not None for p in r.kv_projectors.parameters()):raise AssertionError('Readout loss must not train projectors')
        rt.project(h.detach(),ctx)
        projected_length={str(d):cache.layers[d-1].keys.shape[-2] for d in (16,20,24)}
        next_h,next_ctx=rt.begin(t['tokens'][j:j+1][None,:],cache)
        full=rt.finish(next_h,next_ctx)
        next_loss=kd(full,t,j+1,rt.config['temperature']);next_loss.backward()
    norms={name:float(param.grad.norm()) if param.grad is not None else 0. for name,param in r.kv_projectors.named_parameters()}
    if not all(torch.isfinite(torch.tensor(v)) and v>0 for v in norms.values()):raise AssertionError(f'Projection gradient path failed: {norms}')
    assert all(not p.requires_grad and p.grad is None for p in r.model.parameters())
    detach_cache(cache)
    assert all(getattr(layer,name).grad_fn is None for layer in cache.layers for name in ('keys','values','conv_states','recurrent_states') if getattr(layer,name,None) is not None)
    # Verify next iteration works after truncation, while retaining cache values.
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        h,ctx=rt.begin(t['tokens'][j+1:j+2][None,:],cache);out=rt.finish(h,ctx)
        assert torch.isfinite(out).all()
    # Deployed reject branch must not execute readout_map or any KV projector.
    calls=[];handles=[]
    for name,module in [('readout',r.readout_map),*[(f'P{k}',v) for k,v in r.kv_projectors.items()]]:
        handles.append(module.register_forward_hook(lambda m,a,o,name=name:calls.append(name)))
    saved_threshold=rt.threshold;rt.threshold=None
    try:
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            cache2=initial_cache(rt,t,row);_,exited=rt.adaptive(t['tokens'][j-1:j][None,:],cache2)
        assert not exited and not calls
    finally:
        for handle in handles:handle.remove()
        rt.threshold=saved_threshold
    for p in r.parameters():p.grad=None
    r.requires_grad_(False)
    # Exercise the production training loop with synthetic qualifying labels and a
    # no-op optimizer: gradients and pair scheduling are real, no weights change.
    from unittest.mock import patch
    import numpy as np
    synthetic={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in row.items()}
    synthetic['end']=synthetic['start']+2
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        local=initial_cache(rt,t,row)
        for pos in range(j,j+2):
            h,ctx=rt.begin(t['tokens'][pos-1:pos][None,:],local);z=rt.readout(h).float()
            values,ids=z.topk(8,-1)
            synthetic['ids'][pos]=ids.cpu().numpy()[0];synthetic['logits'][pos]=values.cpu().numpy()[0]
            synthetic['normalizers'][pos]=torch.stack([z.logsumexp(-1),(z/2).logsumexp(-1)],-1).cpu().numpy()[0]
            rt.finish(h,ctx)
    before={name:p.detach().clone() for name,p in r.named_parameters() if name.startswith(('readout_map','kv_projectors','confidence_head'))}
    captured=[]
    class NoUpdate:
        def __init__(self,params,**kwargs):self.params=list(params)
        def step(self):captured.append([float(p.grad.norm()) if p.grad is not None else 0. for p in self.params])
        def zero_grad(self,set_to_none=True):
            for p in self.params:p.grad=None
    class SyntheticData:
        def rows(self,*args,**kwargs):yield synthetic
    with patch('torch.optim.AdamW',NoUpdate):
        loop=module_pass(rt,SyntheticData(),'train',0,2,train=True)
    assert loop['positions']==2 and loop['qualified_candidates']==2 and loop['kv_pairs']==1
    assert len(captured)==2 and all(v>0 for v in captured[1])
    for name,p in r.named_parameters():
        if name in before:assert torch.equal(p,before[name])
    # Deadline reached by projection must still allow the immediate full step.
    from .budget import TokenBudget
    now=[0.];budget=TokenBudget(2,1.,clock=lambda:now[0])
    original_project=rt.project
    def expiring_project(*args,**kwargs):
        original_project(*args,**kwargs);now[0]=2.
    with patch('torch.optim.AdamW',NoUpdate),patch.object(rt,'project',expiring_project):
        bounded=module_pass(rt,SyntheticData(),'train',0,2,train=True,budget=budget)
    assert bounded['positions']==2 and bounded['kv_pairs']==1
    assert bounded['budget']['stop_reason']=='WALL_TIME_LIMIT'
    with patch('torch.optim.AdamW',NoUpdate):
        single=module_pass(rt,SyntheticData(),'train',0,2,train=True,budget=TokenBudget(1,60.))
    assert single['positions']==1 and single['kv_pairs']==0 and single['projection_updates']==0
    for name,p in r.named_parameters():
        if name in before:assert torch.equal(p,before[name])
    r.requires_grad_(False)
    # Bounded end-to-end evaluator/calibrator checks; no confidence training.
    from .evaluation import calibrate_gate,evaluate
    class TinyRealData:
        def rows(self,split,*args,**kwargs):yield next(data.rows(split,count=8))
    original_config=rt.config;original_threshold=rt.threshold
    try:
        rt.config={**original_config,'eval_prompts':1,'decode_lengths':[4]}
        calibration=calibrate_gate(rt,TinyRealData(),1)
        assert calibration['status']=='NO_ELIGIBLE_THRESHOLD'  # only eight calibration samples
        outcome=evaluate(rt,TinyRealData(),1)
        assert outcome['static']['positions']==8 and outcome['static']['exits']==0
        assert outcome['generations'][0]['full']['ids']==outcome['generations'][0]['adaptive']['ids']
        import math
        assert math.isfinite(outcome['static']['top8_tail_kl']) and math.isfinite(outcome['static']['ppl_ratio'])
    finally:
        rt.config=original_config;rt.threshold=original_threshold
    return {'status':'PASS' ,'optimizer_steps':0,'trained_weights_saved':False,
            'native_full_max_logit_errors':errors,'readout_grad_norm':readout_norm,'projection_grad_norms':norms,
            'projected_cache_lengths':projected_length,'base_frozen_and_no_grad':True,'reject_skips_readout_and_projectors':True,
            'pair_graph_check':'Forced synthetic branch only in --verify; production training always uses strict Top8 qualification.',
            'real_position_qualification':{'accepted':bool(actual_good.item()),'overlap':int(overlap.item()),'kl':float(distance.item())},
            'input_alignment':'Next full step consumes teacher completion[j] and supervises stored logits[j+1]',
            'warm_start_exact':rt.initialization,'pair_safe_time_limit':True,'one_position_budget_no_projection':True,
            'state_bias_metrics':False,'bounded_evaluation_check':{'positions':8,'generated_steps':4,'reject_path_identical':True,'calibration_uses_top8_labels':True},'production_pair_loop_no_update_check':{'positions':loop['positions'],'qualified_candidates':loop['qualified_candidates'],'pairs':loop['kv_pairs'],'weights_unchanged':True}}
