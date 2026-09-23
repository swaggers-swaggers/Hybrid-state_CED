"""CPU math, branch and real data alignment tests. No GPU or training."""
import unittest
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ced_distill.losses import teacher_probs,student_log_probs,sparse_kl,proximity
from ced_distill.protocol import pair_action

class DistillTests(unittest.TestCase):
    def targets(self,logits):
        values,ids=logits.topk(8,-1)
        norm=torch.stack([logits.logsumexp(-1),(logits/2).logsumexp(-1)],-1)
        return ids,values,norm
    def test_same_distribution_zero_loss(self):
        torch.manual_seed(1);x=torch.randn(4,17);ids,z,n=self.targets(x)
        self.assertTrue(torch.allclose(sparse_kl(x,ids,z,n),torch.zeros(4),atol=1e-6))
    def test_tail_includes_all_remaining_tokens(self):
        x=torch.arange(17.).reshape(1,17);ids,z,n=self.targets(x)
        coarse=student_log_probs(x,ids).exp();full=x.softmax(-1)
        torch.testing.assert_close(coarse[:,:8],full.gather(-1,ids))
        torch.testing.assert_close(coarse[:,-1],full[:,:9].sum(-1))
    def test_tail_backpropagates(self):
        teacher=torch.arange(17.).reshape(1,17);ids,z,n=self.targets(teacher)
        student=torch.zeros_like(teacher,requires_grad=True)
        sparse_kl(student,ids,z,n).sum().backward()
        self.assertTrue(torch.isfinite(student.grad).all());self.assertGreater(float(student.grad[0,0].abs()),0)
    def test_top1_difference_can_be_trusted(self):
        teacher=torch.tensor([[5.,4.99,4.,3.,2.,1.,0.,-1.,-2.,-3.]])
        student=teacher.clone();student[0,0]=4.99;student[0,1]=5.
        ids,z,n=self.targets(teacher);good,overlap,kl=proximity(student,ids,z,n)
        self.assertNotEqual(teacher.argmax().item(),student.argmax().item());self.assertTrue(good.item());self.assertEqual(overlap.item(),8)
    def test_overlap_alone_not_sufficient(self):
        teacher=torch.arange(17.).reshape(1,17);student=teacher*4
        ids,z,n=self.targets(teacher);good,overlap,kl=proximity(student,ids,z,n)
        self.assertEqual(overlap.item(),8);self.assertGreater(kl.item(),.05);self.assertFalse(good.item())
    def test_temperature_two_matches_coarse_reference(self):
        teacher=torch.arange(12.).reshape(1,12);ids,z,n=self.targets(teacher)
        p=teacher_probs(z,n,2.);q=student_log_probs(teacher,ids,2.).exp()
        torch.testing.assert_close(p,q,atol=1e-6,rtol=1e-5)
    def test_roundoff_tail_is_nonnegative(self):
        z=torch.zeros(1,8);n=torch.tensor([[torch.log(torch.tensor(8.)).item()-1e-6,3.]])
        p=teacher_probs(z,n);self.assertTrue((p>=0).all());torch.testing.assert_close(p.sum(-1),torch.ones(1))
    def test_corrupt_mass_rejected(self):
        with self.assertRaises(ValueError):teacher_probs(torch.zeros(1,8),torch.zeros(1,2))
    def test_pending_always_full_even_if_qualified(self):
        for good in (True,False):self.assertEqual(pair_action(True,good,True),'full_pair')
    def test_rejection_never_projects(self):self.assertEqual(pair_action(False,False,True),'full')
    def test_last_position_never_projects(self):self.assertEqual(pair_action(False,True,False),'full')
    def test_qualified_pair_sequence(self):
        pending=False;actions=[]
        for index in range(5):
            action=pair_action(pending,True,index<4);actions.append(action);pending=action=='exit'
        self.assertEqual(actions,['exit','full_pair','exit','full_pair','full'])

class SparseAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json
        from ced_distill.data import SparseData
        root=Path(__file__).resolve().parents[1]
        if not (root/'data/teacher_top8_20260920/dataset/manifest.json').exists():raise unittest.SkipTest('Local dataset unavailable')
        cls.data=SparseData(root,json.loads((root/'configs/qwen35_08b_top8_distill.json').read_text()))
    def test_partial_response_alignment(self):
        full=next(self.data.rows('train'));part=next(self.data.rows('train',1,2))
        self.assertEqual((part['start'],part['end']),(2,4))
        self.assertTrue((part['tokens'][part['start']-1:part['end']-1]==full['tokens'][1:3]).all())
    def test_cross_response_budget(self):
        full=next(self.data.rows('train'));edge=full['global_end']-1
        rows=list(self.data.rows('train',edge,2))
        self.assertEqual(len(rows),2);self.assertEqual(sum(r['end']-r['start'] for r in rows),2)
        self.assertEqual(rows[0]['global_end'],rows[1]['global_start'])

class BudgetTests(unittest.TestCase):
    def test_sizing_uses_elapsed_time_and_safety_once(self):
        from ced_distill.budget import TokenBudget
        now=[0.];b=TokenBudget(500000,100.,clock=lambda:now[0],sample=10,safety=.5)
        now[0]=10.;self.assertFalse(b.boundary(10));self.assertEqual(b.target,55)
        now[0]=11.;self.assertFalse(b.boundary(20));self.assertEqual(b.target,55)
        self.assertTrue(b.boundary(55));self.assertEqual(b.reason,'TOKEN_TARGET')
    def test_deadline_and_maximum(self):
        from ced_distill.budget import TokenBudget
        now=[0.];b=TokenBudget(20,10.,clock=lambda:now[0],sample=2)
        now[0]=.01;self.assertFalse(b.boundary(2));self.assertEqual(b.target,20)
        now[0]=10.;self.assertTrue(b.boundary(3));self.assertEqual(b.reason,'WALL_TIME_LIMIT')
    def test_confidence_preserves_initialization(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from ced_distill.training import train_confidence
        head=torch.nn.Sequential(torch.nn.Linear(4,1));runner=torch.nn.Module();runner.add_module('confidence_head',head)
        with torch.no_grad():head[0].weight.fill_(.2);head[0].bias.fill_(.3)
        x=torch.ones(4,4);y=torch.ones(4)
        expected=torch.nn.functional.binary_cross_entropy_with_logits(head(x).reshape(-1),y).item()
        rt=SimpleNamespace(runner=runner,config=dict(gate_epochs=1,gate_batch_size=4,gate_lr=.001,gradient_clip=1.,seed=1))
        with patch('ced_distill.training.gate_features',return_value=(x,y)),patch.object(torch.Tensor,'cuda',lambda t,*a,**k:t),patch.object(head[0],'reset_parameters',side_effect=AssertionError('must not reset')):
            result=train_confidence(rt,None)
        self.assertTrue(result['warm_started']);self.assertAlmostEqual(result['initial_dev']['bce'],expected)

if __name__=='__main__':unittest.main()
