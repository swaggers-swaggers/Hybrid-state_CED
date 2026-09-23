"""CPU invariants for shared batches, independent caches and exact credit horizons."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ced_distill.batched import schedule_rollouts,merge_prefixes,lane_where
from ced_distill.losses import sparse_kl

class BatchTests(unittest.TestCase):
    def test_all_64_successors_including_checkpoint_edges(self):
        exits,losses,detach=schedule_rollouts([[True,True]]*132,[132,9],64)
        self.assertEqual([i for i,x in enumerate(exits) if x[0]],[0,65,130])
        self.assertEqual([i for i,x in enumerate(exits) if x[1]],[0])
        self.assertTrue(all(losses[i][0] and not exits[i][0] for i in range(1,65)))
        self.assertFalse(any(detach[i][0] for i in range(64)))
        self.assertTrue(detach[64][0]);self.assertTrue(detach[8][1])
        self.assertEqual(sum(x[1] for x in losses),8)
        self.assertFalse(any(x[1] for x in losses[9:]))
    def test_rejections_and_last_position_do_not_project(self):
        exits,losses,_=schedule_rollouts([[False],[False],[True]],[3],64)
        self.assertFalse(any(map(any,exits)));self.assertFalse(any(map(any,losses)))
    def test_short_response_supervision_is_not_discarded(self):
        exits,losses,_=schedule_rollouts([[False],[True],[True],[True]],[4],64)
        self.assertEqual(exits,[[False],[True],[False],[False]])
        self.assertEqual(losses,[[False],[False],[True],[True]])
    def test_merged_cache_padding_and_independence(self):
        a=SimpleNamespace(layers=[SimpleNamespace(keys=torch.ones(1,2,3,4),values=torch.ones(1,2,3,4)),
                                 SimpleNamespace(conv_states=torch.ones(1,8,4),recurrent_states=torch.ones(1,2,4,4))])
        b=SimpleNamespace(layers=[SimpleNamespace(keys=torch.full((1,2,5,4),2.),values=torch.full((1,2,5,4),2.)),
                                 SimpleNamespace(conv_states=torch.full((1,8,4),2.),recurrent_states=torch.full((1,2,4,4),2.))])
        c=merge_prefixes([a,b],[3,5])
        self.assertTrue((c.layers[0].keys[0,:,:2]==0).all());self.assertTrue((c.layers[0].keys[0,:,2:]==1).all())
        self.assertTrue((c.layers[1].recurrent_states[1]==2).all())
        c.layers[0].keys.zero_();self.assertTrue((a.layers[0].keys==1).all());self.assertTrue((b.layers[0].keys==2).all())
    def test_one_lane_detach_does_not_cut_other_gradient(self):
        a=torch.tensor([[2.],[3.]],requires_grad=True)
        lane_where(torch.tensor([True,False]),a.detach(),a).sum().backward()
        torch.testing.assert_close(a.grad,torch.tensor([[0.],[1.]]))
    def test_shared_batch_gradient_is_valid_token_weighted_sum(self):
        torch.manual_seed(31);h=torch.randn(2,5,4);vocab=torch.randn(4,16)
        teacher=torch.randn(2,5,16);values,ids=teacher.topk(8,-1)
        normals=torch.stack([teacher.logsumexp(-1),(teacher/2).logsumexp(-1)],-1)
        valid=torch.arange(5)[None,:]<torch.tensor([5,3])[:,None]
        w=torch.randn(4,4,requires_grad=True)
        sparse_kl(h[valid]@w@vocab,ids[valid],values[valid],normals[valid]).mean().backward();combined=w.grad.clone();w.grad=None
        for lane,n in enumerate((5,3)):
            (sparse_kl(h[lane,:n]@w@vocab,ids[lane,:n],values[lane,:n],normals[lane,:n]).sum()/8).backward()
        torch.testing.assert_close(w.grad,combined,atol=2e-6,rtol=1e-5)

if __name__=='__main__':unittest.main()
