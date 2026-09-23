"""CPU checks for sparse teacher storage, sampling alignment and disk limits."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ced_training.topk_data import (article_split,sparse_and_sample,accept_responses,
                                    supervision_mask,validate_shard,check_storage)


class SparseDataTests(unittest.TestCase):
    def test_raw_logits_and_full_normalizers(self):
        raw=torch.tensor([[7.,2.,9.,1.,8.,3.,6.,4.,5.,0.]]).repeat(3,1)
        g=torch.Generator().manual_seed(7)
        token,ids,values,norm,chosen=sparse_and_sample(raw,.8,.95,g)
        torch.testing.assert_close(values,raw.gather(1,ids))
        torch.testing.assert_close(norm[:,0],raw.logsumexp(-1))
        torch.testing.assert_close(norm[:,1],(raw/2).logsumexp(-1))
        torch.testing.assert_close(chosen,raw.gather(1,token[:,None]).squeeze(-1))
        self.assertEqual(ids.shape,(3,8))
        self.assertTrue(((values-norm[:,0,None]).exp().sum(-1)<1).all())

    def test_nucleus_preserves_first_crossing(self):
        raw=torch.log(torch.tensor([[.4,.3,.2,.1,1e-12,1e-12,1e-12,1e-12]]).repeat(2048,1))
        token,*_=sparse_and_sample(raw,1.,.5,torch.Generator().manual_seed(4))
        self.assertEqual(set(token.tolist()),{0,1})

    def test_deterministic_sampling(self):
        raw=torch.randn(10,17)
        a=sparse_and_sample(raw,.8,.95,torch.Generator().manual_seed(8))
        b=sparse_and_sample(raw,.8,.95,torch.Generator().manual_seed(8))
        for x,y in zip(a,b):self.assertTrue(torch.equal(x,y))

    def test_sparse_record_alignment_and_mask(self):
        raw=torch.arange(12.).repeat(3,4,1)
        token,ids,values,norm,chosen=sparse_and_sample(raw.reshape(-1,12),.8,.95,torch.Generator().manual_seed(4))
        lengths=np.array([4,4,4]);accepted=np.array([True,True,True])
        a={'tokens':token.reshape(3,4).numpy(),'lengths':lengths,'accepted':accepted,
           'loss_mask':supervision_mask(lengths,accepted,4),'top8_ids':ids.reshape(3,4,8).numpy(),
           'top8_logits':values.reshape(3,4,8).numpy(),'logsumexp':norm.reshape(3,4,2).numpy(),
           'sampled_logit':chosen.reshape(3,4).numpy(),'prompt_ids':np.array([9,9,9])}
        self.assertEqual(validate_shard(a,vocab=12)['effective_targets'],9)
        a['loss_mask'][0,0]=1
        with self.assertRaises(AssertionError):validate_shard(a,vocab=12)

    def test_dedup_keeps_two_or_three_only(self):
        tokens=np.array([[1,2,3],[1,2,3],[4,5,6],[7,8,9],[7,8,9],[7,8,9]])
        seen=set();a=accept_responses(tokens,np.array([3]*6),3,seen)
        self.assertEqual(a.tolist(),[True,False,True,False,False,False])
        self.assertEqual(len(seen),2)
        self.assertFalse(accept_responses(tokens[:3],np.array([3]*3),3,seen).any())

    def test_first_output_and_padding_never_supervise(self):
        m=supervision_mask(np.array([1,3,5]),np.array([True,True,False]),5)
        np.testing.assert_array_equal(m,[[0,0,0,0,0],[0,1,1,0,0],[0,0,0,0,0]])

    def test_article_group_is_stable(self):
        self.assertEqual(article_split('example title'),article_split('example title'))
        self.assertEqual(set(article_split(str(x)) for x in range(10000)),{'train','gate','dev','calibration','test'})

    def test_storage_cap_counts_existing_files(self):
        with tempfile.TemporaryDirectory() as name:
            p=Path(name);(p/'a').write_bytes(b'0'*60)
            self.assertEqual(check_storage(p,30,100,0),60)
            with self.assertRaises(RuntimeError):check_storage(p,41,100,0)

    def test_free_space_reserve_before_write(self):
        from collections import namedtuple
        usage=namedtuple('usage','total used free')(100,50,50)
        with tempfile.TemporaryDirectory() as name,patch('shutil.disk_usage',return_value=usage):
            with self.assertRaises(RuntimeError):check_storage(name,20,1000,40)

if __name__=='__main__':unittest.main()
