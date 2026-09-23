"""Stream audited sparse records without materializing full-vocabulary targets."""
import json
from pathlib import Path
import numpy as np
from ced_training.protocol import sha256
from ced_training.topk_data import validate_shard

class SparseData:
    def __init__(self,root,config):
        self.path=Path(root)/config['data_path'];self.digest=sha256(self.path/'manifest.json')
        if self.digest!=config['manifest_sha256']: raise ValueError('Data manifest changed')
        self.manifest=json.loads((self.path/'manifest.json').read_text())
        m=self.manifest
        if m['status']!='COMPLETE_DATA_ONLY_NO_TRAINING' or m['config']['stored_topk']!=8: raise ValueError('Incomplete/non-Top8 data')
        if json.loads((self.path/'audit.json').read_text())['status']!='PASS': raise ValueError('Missing data audit')
        self.checked=set()
    def rows(self,split,start=0,count=None,max_responses=None):
        total=self.manifest['splits'][split]['effective_targets']
        end=total if count is None else start+count
        if not 0<=start<end<=total: raise ValueError('Invalid sparse data interval')
        cursor=0;responses=0
        for item in self.manifest['splits'][split]['shards']:
            path=self.path/item['file']
            if path not in self.checked:
                if path.stat().st_size!=item['bytes'] or sha256(path)!=item['sha256']: raise ValueError(f'Corrupt shard: {path}')
                self.checked.add(path)
            with np.load(path,allow_pickle=False) as a:
                for row,accepted in enumerate(a['accepted']):
                    if not accepted: continue
                    length=int(a['lengths'][row]);size=length-1
                    first,last=max(cursor,start),min(cursor+size,end)
                    if first<last:
                        if max_responses is not None and responses>=max_responses:return
                        j0,j1=1+first-cursor,1+last-cursor
                        mask=a['loss_mask'][row,:length]
                        if mask[0] or not np.all(mask[1:]==1):raise ValueError('Unexpected supervision mask')
                        yield {'prompt':a['prompt_tokens'][row].copy(),'tokens':a['tokens'][row,:length].copy(),
                               'ids':a['top8_ids'][row,:length].copy(),'logits':a['top8_logits'][row,:length].copy(),
                               'normalizers':a['logsumexp'][row,:length].copy(),'sampled_logit':a['sampled_logit'][row,:length].copy(),
                               'prompt_id':int(a['prompt_ids'][row]),'answer_index':int(a['answer_index'][row]),
                               'start':j0,'end':j1,'global_start':first,'global_end':last}
                        responses+=1
                    cursor+=size
                    if cursor>=end:return
        if cursor<end:raise ValueError('Sparse data exhausted before declared count')
