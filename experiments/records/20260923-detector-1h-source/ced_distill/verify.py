"""Real-model correctness/gradient checks only; no timing and no parameter updates."""
import copy
from unittest.mock import patch
import numpy as np
import torch
from .training import tensors,initial_cache,module_pass
from .losses import proximity,sparse_kl
from .evaluation import evaluate


def verify(rt,data):
    row=next(data.rows('train',count=34));t=tensors(row);j=row['start'];r=rt.runner
    if row['end']-j!=34:raise AssertionError('Verification needs 34 consecutive positions')
    before={name:p.detach().clone() for name,p in r.named_parameters()
            if name.startswith(('readout_map','kv_projectors','confidence_head'))}
    with torch.no_grad():
        a=initial_cache(rt,t,row);b=copy.deepcopy(a);manual=[];hidden=[]
        for pos in range(j,row['end']):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                h,ctx=rt.begin(t['tokens'][pos-1:pos][None,:],a);hidden.append(h.clone())
                if pos<j+2:
                    x=rt.finish(h,ctx);native=r.native(t['tokens'][pos-1:pos][None,:],b)[:,-1,:]
                    torch.testing.assert_close(x,native,atol=.0625,rtol=0);manual.append(float((x-native).abs().max()))
        sequential=torch.cat(hidden,1)
        chunk_cache=initial_cache(rt,t,row);chunked=[];chunk_logits=[]
        for pos in range(j,row['end'],rt.config['readout_chunk_size']):
            end=min(row['end'],pos+rt.config['readout_chunk_size'])
            with torch.autocast('cuda',dtype=torch.bfloat16):
                h,ctx=rt.begin(t['tokens'][pos-1:end-1][None,:],chunk_cache)
                chunked.append(h);chunk_logits.append(rt.readout_all(h).float())
        chunked=torch.cat(chunked,1);z=torch.cat(chunk_logits)
        # BF16 matrix kernels and recurrent/chunk reduction orders can differ.
        relative=float((chunked.float()-sequential.float()).norm()/sequential.float().norm())
        if relative>.015:raise AssertionError(f'Chunk hidden relative error {relative}')
        with torch.autocast('cuda',dtype=torch.bfloat16):reference=rt.readout_all(sequential)
        output_kl_difference=float((sparse_kl(z,t['ids'][j:row['end']],t['logits'][j:row['end']],t['normalizers'][j:row['end']])-
                                   sparse_kl(reference,t['ids'][j:row['end']],t['logits'][j:row['end']],t['normalizers'][j:row['end']])).abs().max())
        if output_kl_difference>.05:raise AssertionError('Chunking changed detector loss excessively')
        # Change only future tokens; an earlier prediction must stay invariant.
        ca=initial_cache(rt,t,row);cb=copy.deepcopy(ca);tokens=t['tokens'][j-1:j+7].clone()[None,:];altered=tokens.clone();altered[:,4:]=17
        with torch.autocast('cuda',dtype=torch.bfloat16):ha,_=rt.begin(tokens,ca);hb,_=rt.begin(altered,cb)
        future_error=float((ha[:,:4]-hb[:,:4]).abs().max())
        torch.testing.assert_close(ha[:,:4],hb[:,:4],atol=.015625,rtol=.001)
    del a,b,ca,cb,chunk_cache,hidden,chunked,sequential,reference,chunk_logits
    synthetic={key:(value.copy() if isinstance(value,np.ndarray) else value) for key,value in row.items()}
    values,ids=z.topk(8,-1)
    synthetic['ids'][j:row['end']]=ids.cpu().numpy();synthetic['logits'][j:row['end']]=values.cpu().numpy()
    synthetic['normalizers'][j:row['end']]=torch.stack([z.logsumexp(-1),(z/2).logsumexp(-1)],-1).cpu().numpy()
    # First position fails the OR rule; other 33 qualify. This places an exit at
    # offset 31 and its full successor at offset 32, across a 32-position chunk edge.
    excluded=set(ids[0].tolist());other=[x for x in range(32) if x not in excluded][:8]
    synthetic['ids'][j]=other
    captured=[]
    class NoUpdate:
        def __init__(self,params,**kwargs):self.params=list(params)
        def step(self):captured.append([float(p.grad.norm()) if p.grad is not None else 0. for p in self.params])
        def zero_grad(self,set_to_none=True):
            for p in self.params:p.grad=None
    class SyntheticData:
        def rows(self,*args,**kwargs):yield synthetic
    with patch('torch.optim.AdamW',NoUpdate):loop=module_pass(rt,SyntheticData(),'train',0,34,train=True)
    if loop['positions']!=34 or loop['kv_pairs']!=16 or loop['qualified_candidates']!=33:raise AssertionError(loop)
    if loop['readout_batches']!=2 or loop['shallow_batches']!=2:raise AssertionError('Expected chunked matrix operations')
    if len(captured)!=2 or not all(v>0 and np.isfinite(v) for values in captured for v in values):raise AssertionError(captured)
    assert all(not p.requires_grad and p.grad is None for p in r.model.parameters())
    for name,p in r.named_parameters():
        if name in before:assert torch.equal(p,before[name])
    # The head-only evaluator must be identical for accept-all / reject-all settings.
    class TinyRealData:
        def rows(self,split,*args,**kwargs):yield next(data.rows(split,count=8))
    threshold=rt.threshold
    try:
        rt.threshold=1.1;reject=evaluate(rt,TinyRealData(),1)
        rt.threshold=-1.;accept=evaluate(rt,TinyRealData(),1)
        assert reject==accept and reject['detector']['positions']==8
        assert reject['detector']['output_source']=='block12_readout_at_every_position'
    finally:rt.threshold=threshold
    return {'status':'PASS','optimizer_steps':0,'trained_weights_saved':False,'cost_benchmark_executed':False,
            'warm_start_exact':rt.initialization,'native_full_max_logit_errors':manual,
            'chunk_hidden_relative_error':relative,'chunk_max_detector_kl_difference':output_kl_difference,
            'future_token_change_prefix_max_error':future_error,'unconditional_detector_evaluation':True,
            'production_loop':loop,'readout_gradient_norms':captured[0],'projection_gradient_norms':captured[1],
            'frozen_base_no_grad':True,'all_auxiliary_weights_unchanged':True,
            'verification_labels':'Synthetic OR-qualified positions exercise pair and chunk boundaries; not a training-quality result.'}
