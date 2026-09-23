"""Bounded real-model checks: no optimizer updates, trained checkpoints or cost tests."""
import copy
from unittest.mock import patch
import numpy as np
import torch
from .training import tensors,prefix_tokens,module_pass
from .batched import ResponseBatch,projection_objective,deep_step,flatten_cache
from .evaluation import evaluate


def verify(rt,data):
    source=list(data.rows('train',max_responses=2))
    if len(source)!=2:raise AssertionError('Two distinct responses required')
    rows=[]
    for lane,row in enumerate(source):
        row=copy.deepcopy(row);row['start']+=lane+1;row['end']=row['start']+66+lane
        if row['end']>len(row['tokens']):raise AssertionError('Verification response too short')
        rows.append(row)
    runner=rt.runner
    before={name:p.detach().clone() for name,p in runner.named_parameters()
            if name.startswith(('readout_map','kv_projectors','confidence_head'))}
    config=rt.config;rt.config={**config,'projection_horizon':64,'checkpoint_steps':8}
    try:
        batch=ResponseBatch(rt,rows);hidden,_=batch.shallow(rt)
        relative=[]
        with torch.no_grad():
            for lane,row in enumerate(rows):
                t=tensors(row);cache=rt.cache()
                with torch.autocast('cuda',dtype=torch.bfloat16):rt.begin(prefix_tokens(t,row),cache)
                individual=[]
                for start in range(row['start'],row['end'],config['readout_chunk_size']):
                    end=min(row['end'],start+config['readout_chunk_size'])
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        h,_=rt.begin(t['tokens'][start-1:end-1][None,:],cache)
                    individual.append(h)
                ref=torch.cat(individual,1)[0].float();candidate=hidden[lane,:len(ref)].float()
                error=float((candidate-ref).norm()/ref.norm());relative.append(error)
                if error>.025:raise AssertionError(f'Batched/padded shallow mismatch {error}')
            # Causal/padding isolation: changing the other lane cannot affect lane 0.
            modified=ResponseBatch(rt,rows);modified.inputs[1]=17
            alternate,_=modified.shallow(rt)
            torch.testing.assert_close(hidden[0],alternate[0],atol=0,rtol=0)
            del modified,alternate,individual,ref,candidate,cache
        # Explicit padded deep attention must match two independent full executions.
        full_errors=[]
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            from .batched import cache_copy
            full=cache_copy(batch.template)
            for layer in full.layers[12:]:
                if getattr(layer,'conv_states',None) is not None:layer.conv_states=layer.conv_states.clone()
            z=rt.finish(hidden[:,:1],batch.context(rt,hidden[:,:1],full,0,1)).float()
            for lane,row in enumerate(rows):
                t=tensors(row);cache,_=rt.prefill(prefix_tokens(t,row))
                ref=runner.native(t['tokens'][row['start']-1:row['start']][None,:],cache)[:,-1,:].float()
                error=float((z[lane:lane+1]-ref).norm()/ref.norm());full_errors.append(error)
                if error>.025:raise AssertionError(f'Batched deep output mismatch {error}')
            del full,z,ref,cache
        runner.kv_projectors.requires_grad_(True)
        labels=[[step==0]*2 for step in range(batch.steps)]
        # A loss ONLY at the 64th successor must reach all six projection matrices.
        terminal,stats=projection_objective(rt,batch,hidden,labels,loss_steps={64})
        terminal.backward()
        far_grad=[float(p.grad.norm()) for p in runner.kv_projectors.parameters()]
        if not all(x>0 and np.isfinite(x) for x in far_grad):raise AssertionError(far_grad)
        if stats['kv_supervised_positions']!=128:raise AssertionError(stats)
        runner.zero_grad(set_to_none=True);del terminal
        # The 65th successor is outside the chosen horizon.
        outside,_=projection_objective(rt,batch,hidden,labels,loss_steps={65},train=False,use_checkpoint=False)
        if float(outside.detach())!=0:raise AssertionError('Horizon leaked')
        del outside
        # Anchor cache contents and GDN hold, not numerical GDN-deviation metrics.
        original=flatten_cache(batch.template,batch.spec)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            anchor,_=deep_step(rt,batch,batch.template,hidden[:,:1],0,
                torch.tensor([True,True],device='cuda'),torch.tensor([False,False],device='cuda'),
                torch.tensor([False,False],device='cuda'),False)
        for (index,name),old in zip(batch.spec,original):
            current=getattr(anchor.layers[index],name)
            if name in ('keys','values'):
                assert current.shape[-2]==old.shape[-2]+1
                torch.testing.assert_close(current[...,:-1,:],old,atol=0,rtol=0)
            else:torch.testing.assert_close(current,old,atol=0,rtol=0)
            torch.testing.assert_close(getattr(batch.template.layers[index],name),old,atol=0,rtol=0)
        del anchor,original,batch,hidden
        # Recompute and ordinary autograd must agree across multiple segments.
        short=[{**row,'end':row['start']+6-lane} for lane,row in enumerate(rows)]
        rt.config={**rt.config,'checkpoint_steps':2}
        batch=ResponseBatch(rt,short);hidden,_=batch.shallow(rt);labels=[[i==0]*2 for i in range(batch.steps)]
        norms=[];gradients=[];loss_values=[]
        for checkpointed in (False,True):
            loss,_=projection_objective(rt,batch,hidden,labels,use_checkpoint=checkpointed)
            loss.backward();gradients.append([p.grad.detach().clone() for p in runner.kv_projectors.parameters()])
            loss_values.append(float(loss.detach()));runner.zero_grad(set_to_none=True);del loss
        for plain,recomputed in zip(*gradients):
            torch.testing.assert_close(plain,recomputed,rtol=1e-5,atol=1e-6)
            norms.append(float((plain-recomputed).norm()))
        del gradients,batch,hidden
        # Real production loop: two rows, unequal lengths, forced qualifying labels,
        # gradients captured after combined normalization but step intentionally disabled.
        captured=[]
        class NoUpdate:
            def __init__(self,params,**kwargs):self.params=list(params)
            def step(self):captured.append([float(p.grad.norm()) if p.grad is not None else 0. for p in self.params])
            def zero_grad(self,set_to_none=True):
                for p in self.params:p.grad=None
        class TinyData:
            def rows(self,*args,**kwargs):yield from short
        from .losses import proximity
        def accept(logits,ids,minimum_overlap):
            _,overlap,top1=proximity(logits,ids,minimum_overlap)
            return torch.ones_like(top1),overlap,top1
        with patch('torch.optim.AdamW',NoUpdate),patch('ced_distill.batched.proximity',accept):
            production=module_pass(rt,TinyData(),'train',0,11,train=True)
        if production['positions']!=11 or production['kv_projections']!=2 or production['kv_supervised_positions']!=9:raise AssertionError(production)
        if production['maximum_response_batch']!=2 or production['readout_updates']!=1 or production['projection_updates']!=1:raise AssertionError(production)
        if len(captured)!=2 or not all(v>0 and np.isfinite(v) for group in captured for v in group):raise AssertionError(captured)
        # Stress the production graph with complete, differently sized responses.
        rt.config={**config,'projection_horizon':64,'checkpoint_steps':8}
        long_rows=[{**row,'start':1+lane} for lane,row in enumerate(source)]
        long_count=sum(row['end']-row['start'] for row in long_rows)
        class FullData:
            def rows(self,*args,**kwargs):yield from long_rows
        with patch('torch.optim.AdamW',NoUpdate),patch('ced_distill.batched.proximity',accept):
            full_length=module_pass(rt,FullData(),'train',0,long_count,train=True)
        if full_length['maximum_response_batch']!=2 or full_length['projection_updates']!=1:raise AssertionError(full_length)
        if not all(v>0 and np.isfinite(v) for group in captured[2:] for v in group):raise AssertionError(captured)
        assert all(not p.requires_grad and p.grad is None for p in runner.model.parameters())
        for name,p in runner.named_parameters():
            if name in before:assert torch.equal(p,before[name])
        class TinyRealData:
            def rows(self,split,*args,**kwargs):yield next(data.rows(split,count=8))
        threshold=rt.threshold
        try:
            rt.threshold=1.1;reject=evaluate(rt,TinyRealData(),1)
            rt.threshold=-1.;accept=evaluate(rt,TinyRealData(),1)
            assert reject==accept and reject['detector']['positions']==8
        finally:rt.threshold=threshold
        return dict(status='PASS',optimizer_steps=0,trained_weights_saved=False,cost_benchmark_executed=False,
                    warm_start_exact=rt.initialization,batched_shallow_relative_errors=relative,
                    batched_deep_output_relative_errors=full_errors,full_length_production_loop=full_length,
                    other_lane_change_has_no_effect=True,projection_successor_64_gradient_norms=far_grad,
                    successor_65_outside_horizon=True,checkpoint_vs_plain_gradient_difference_norms=norms,
                    checkpoint_vs_plain_losses=loss_values,anchor_gdn_held_and_input_cache_unchanged=True,
                    production_loop=production,readout_gradient_norms=captured[0],projection_gradient_norms=captured[1],
                    frozen_base_no_grad=True,all_auxiliary_weights_unchanged=True,unconditional_detector_evaluation=True,
                    peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                    verification_labels='Synthetic qualifying anchors for gradient/branch coverage, not quality evaluation.')
    finally:
        rt.config=config;runner.requires_grad_(False);runner.zero_grad(set_to_none=True)
