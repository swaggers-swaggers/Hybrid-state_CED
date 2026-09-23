#!/usr/bin/env python3
"""Generate original-teacher tokens and Top-8 raw logits only; never train."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ced_training.protocol import sha256
from ced_training.topk_data import check_storage
STORAGE_ROOT = None
MAXIMUM_BYTES = 2_000_000_000
RESERVE_BYTES = 20_000_000_000


def guard(path, pending_bytes):
    if STORAGE_ROOT is None or not Path(path).resolve().is_relative_to(STORAGE_ROOT):
        raise RuntimeError("Artifact outside the declared storage budget root")
    check_storage(STORAGE_ROOT, pending_bytes, MAXIMUM_BYTES, RESERVE_BYTES)


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    guard(path, len(payload.encode()))
    temp.write_text(payload)
    temp.replace(path)


def prepare_prompts(config, output):
    import numpy as np
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from ced_training.corpus import complete_articles, title_key
    from ced_training.topk_data import article_split
    output.mkdir(parents=True, exist_ok=False)
    fresh = ROOT / config['fresh_corpus']; prior = ROOT / config['prior_corpus']
    fresh_m = json.loads((fresh/'manifest.json').read_text())
    prior_m = json.loads((prior/'manifest.json').read_text())
    assert sha256(fresh/'articles.jsonl') == fresh_m['files']['articles.jsonl']['sha256']
    articles = [json.loads(s) for s in (fresh/'articles.jsonl').read_text().splitlines()]
    cutoff = max(a['source_last_row'] for a in articles)
    tokenizer_path = ROOT / config['model_path']
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    assert tokenizer.eos_token_id == fresh_m['eos_token_id']
    for name, digest in fresh_m['tokenizer_files'].items():
        assert sha256(tokenizer_path/name) == digest
    excluded = set()
    for directory, files in ((prior, {x['file']:x for x in prior_m['splits'].values()}),
                             (fresh, {'train.int32.bin':fresh_m['files']['train.int32.bin']})):
        for filename, info in files.items():
            path = directory/filename
            assert sha256(path) == info['sha256']
            arr = np.memmap(path, dtype='<i4', mode='r').reshape(-1,256)
            excluded.update(hashlib.sha256(row.tobytes()).digest() for row in arr)
    shards = [Path(x['path']) for x in fresh_m['source_files']]
    for p, info in zip(shards, fresh_m['source_files']):
        assert sha256(p) == info['sha256']
    def rows():
        index = 0
        for shard in shards:
            for batch in pq.ParquetFile(shard).iter_batches(batch_size=4096, columns=['text']):
                for value in batch.column(0).to_pylist():
                    yield index, value
                    index += 1
    prompts, metadata, seen_titles, seen_prompts = [], [], set(), set()
    counts = dict.fromkeys(config['target_effective_tokens'], 0)
    skips = {'consumed_prefix_or_boundary':0,'repeated_title':0,'short_article':0,'repeated_prompt':0}
    for article in complete_articles(rows()):
        key = title_key(article['title'])
        repeated = key in seen_titles
        seen_titles.add(key)
        if article['first_row'] <= cutoff:
            skips['consumed_prefix_or_boundary'] += 1; continue
        if repeated:
            skips['repeated_title'] += 1; continue
        ids = []
        for text in article['texts']:
            ids.extend(tokenizer.encode(text, add_special_tokens=False)); ids.append(tokenizer.eos_token_id)
            if len(ids) >= config['prompt_length']:
                break
        if len(ids) < config['prompt_length']:
            skips['short_article'] += 1; continue
        prompt = np.asarray(ids[:config['prompt_length']], dtype='<i4')
        digest = hashlib.sha256(prompt.tobytes()).digest()
        if digest in excluded or digest in seen_prompts:
            skips['repeated_prompt'] += 1; continue
        seen_prompts.add(digest)
        split = article_split(key)
        metadata.append({'prompt_id':len(prompts),'article_key':key,'title':article['title'],'split':split,
                         'source_first_row':article['first_row'],'source_last_row':article['last_row'],
                         'prompt_sha256':digest.hex()})
        prompts.append(prompt); counts[split] += 1
    for split, target in config['target_effective_tokens'].items():
        if counts[split]*config['answers_per_prompt']*(config['max_new_tokens']-1) < target:
            raise RuntimeError(f'Insufficient prompt capacity for {split}')
    guard(output, len(prompts)*config['prompt_length']*4 + sum(len(json.dumps(r,ensure_ascii=False).encode())+1 for r in metadata) + 1_000_000)
    np.stack(prompts).astype('<i4').tofile(output/'prompts.int32.bin')
    with (output/'articles.jsonl').open('w') as f:
        for row in metadata:
            f.write(json.dumps(row, ensure_ascii=False)+'\n')
    m = {'status':'COMPLETE_PROMPTS_ONLY_NO_GENERATION_OR_TRAINING','count':len(prompts),'counts':counts,
         'prompt_length':config['prompt_length'],'dataset_revision':fresh_m['dataset_revision'],
         'model_revision':prior_m['model_revision'],'model_weights_sha256':prior_m['weights_etag_sha256'],
         'source_files':fresh_m['source_files'],'cutoff_source_row':cutoff,
         'fresh_manifest_sha256':sha256(fresh/'manifest.json'),'prior_manifest_sha256':sha256(prior/'manifest.json'),
         'tokenizer_files':fresh_m['tokenizer_files'],'skips':skips,
         'split_policy':'One unique article and one prompt per article; deterministic title hash; all answers stay in that split.',
         'freshness':'Entire source prefix used by prior natural corpora excluded, plus repeated titles and exact cached windows.',
         'files':{p.name:{'bytes':p.stat().st_size,'sha256':sha256(p)} for p in (output/'prompts.int32.bin',output/'articles.jsonl')}}
    write_json(output/'manifest.json',m)
    print(json.dumps({'output':str(output),'counts':counts,'status':m['status']},ensure_ascii=False),flush=True)


def audit(output):
    import numpy as np
    from ced_training.topk_data import validate_shard
    m = json.loads((output/'manifest.json').read_text())
    pool = Path(m['prompt_pool']); pm = json.loads((pool/'manifest.json').read_text())
    assert sha256(pool/'manifest.json') == m['prompt_manifest_sha256']
    for name,info in pm['files'].items():
        assert sha256(pool/name)==info['sha256'] and (pool/name).stat().st_size==info['bytes']
    records = [json.loads(s) for s in (pool/'articles.jsonl').read_text().splitlines()]
    prompts = np.memmap(pool/'prompts.int32.bin',dtype='<i4',mode='r').reshape(-1,m['config']['prompt_length'])
    assigned, seen_generated = {}, set()
    totals = {}
    for split, info in m['splits'].items():
        counts = {'stored_generated_tokens':0,'effective_targets':0,'accepted_responses':0,'shards':0,'top8_mass_t1_sum':0.,'top8_mass_t1_min':1.,'top8_mass_t1_max':0.}
        for item in info['shards']:
            path=output/item['file']
            assert path.stat().st_size==item['bytes'] and sha256(path)==item['sha256']
            with np.load(path,allow_pickle=False) as arrays:
                a={k:arrays[k] for k in arrays.files}
            assert np.array_equal(a['prompt_tokens'],prompts[a['prompt_ids']])
            stats=validate_shard(a,eos_ids=m['eos_token_ids'])
            assert stats['effective_targets']==item['effective_targets']
            assert len(a['prompt_ids'])%3==0
            for start in range(0,len(a['prompt_ids']),3):
                assert len(set(a['prompt_ids'][start:start+3].tolist()))==1
                assert a['answer_index'][start:start+3].tolist()==[0,1,2]
                pid=int(a['prompt_ids'][start]); article=records[pid]['article_key']
                assert records[pid]['split']==split and article not in assigned
                assigned[article]=split
            for row,length in enumerate(a['lengths']):
                if a['accepted'][row]:
                    digest=hashlib.sha256(a['tokens'][row,:length].astype('<i4').tobytes()).hexdigest()
                    assert digest not in seen_generated
                    seen_generated.add(digest)
            for key in ('stored_generated_tokens','effective_targets','accepted_responses','top8_mass_t1_sum'):
                counts[key]+=stats[key]
            counts['top8_mass_t1_min']=min(counts['top8_mass_t1_min'],stats['top8_mass_t1_min'])
            counts['top8_mass_t1_max']=max(counts['top8_mass_t1_max'],stats['top8_mass_t1_max'])
            counts['shards']+=1
        assert counts['effective_targets']==info['effective_targets']>=m['config']['target_effective_tokens'][split]
        counts['mean_top8_mass_t1']=counts.pop('top8_mass_t1_sum')/counts['stored_generated_tokens']
        totals[split]=counts
    result={'status':'PASS','splits':totals,'unique_articles_used':len(assigned),
            'checks':'SHA256, exact quota, prompt/article split isolation, duplicate answers, token-logit alignment, raw-logit finiteness/order, T1/T2 mass, EOS, first-token/padding/rejected-answer masks'}
    write_json(output/'audit.json',result)
    return result


def replay(model, output, max_steps=16):
    import numpy as np
    import torch
    from transformers import DynamicCache
    m=json.loads((output/'manifest.json').read_text()); checks=[]
    for split,info in m['splits'].items():
        item=info['shards'][0]
        with np.load(output/item['file'],allow_pickle=False) as a:
            prompts=torch.from_numpy(a['prompt_tokens'].astype('int64')).cuda()
            tokens=torch.from_numpy(a['tokens'].astype('int64')).cuda()
            cache=DynamicCache(config=model.config)
            inputs=prompts; maximum_error=0.
            for pos in range(min(max_steps,tokens.shape[1])):
                hidden=model.model.language_model(input_ids=inputs,past_key_values=cache,use_cache=True).last_hidden_state
                logits=model.lm_head(hidden[:,-1,:]).float()
                active=torch.from_numpy(a['lengths']>pos).cuda()
                ids=torch.from_numpy(a['top8_ids'][:,pos].astype('int64')).cuda()
                saved=torch.from_numpy(a['top8_logits'][:,pos]).cuda()
                torch.testing.assert_close(logits.gather(-1,ids)[active],saved[active],atol=.0625,rtol=0)
                sorted_values=logits.sort(-1,descending=True).values[:,:8]
                torch.testing.assert_close(sorted_values[active],saved[active],atol=.0625,rtol=0)
                saved_z=torch.from_numpy(a['logsumexp'][:,pos]).cuda()
                for col,t in enumerate((1.,2.)):
                    torch.testing.assert_close(torch.logsumexp(logits/t,-1)[active],saved_z[:,col][active],atol=.0625,rtol=0)
                generated_logit=logits.gather(-1,tokens[:,pos,None]).squeeze(-1)
                torch.testing.assert_close(generated_logit[active],torch.from_numpy(a['sampled_logit'][:,pos]).cuda()[active],atol=.0625,rtol=0)
                if active.any(): maximum_error=max(maximum_error,float((logits.gather(-1,ids)[active]-saved[active]).abs().max()))
                inputs=tokens[:,pos,None]
        checks.append({'split':split,'file':item['file'],'steps':min(max_steps,tokens.shape[1]),'max_logit_error':maximum_error})
        del cache
    result={'status':'PASS','checks':checks,'meaning':'Full original model replay on stored prefixes, including first output; no training.'}
    write_json(output/'replay_audit.json',result)
    return result


def generate(config,pool,output,smoke=False):
    import numpy as np
    import torch
    from transformers import Qwen3_5ForConditionalGeneration,DynamicCache,__version__
    from ced_training.topk_data import sparse_and_sample,accept_responses,supervision_mask
    assert __version__=='5.12.1' and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    pm=json.loads((pool/'manifest.json').read_text())
    assert pm['status']=='COMPLETE_PROMPTS_ONLY_NO_GENERATION_OR_TRAINING'
    for name,info in pm['files'].items(): assert sha256(pool/name)==info['sha256']
    model_path=ROOT/config['model_path']; weights=list(model_path.glob('*.safetensors'))
    assert len(weights)==1 and sha256(weights[0])==pm['model_weights_sha256']
    for name,digest in pm['tokenizer_files'].items(): assert sha256(model_path/name)==digest
    config=dict(config)
    if smoke:
        config['max_new_tokens']=32
        config['target_effective_tokens']={k:100 for k in config['target_effective_tokens']}
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic(); torch.manual_seed(config['seed']); torch.cuda.manual_seed_all(config['seed'])
    hashes={name:sha256(ROOT/name) for name in ('ced_training/topk_data.py','scripts/generate_top8_data.py')}
    write_json(output/'config.json',config)
    try:
        model=Qwen3_5ForConditionalGeneration.from_pretrained(model_path,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
        model.requires_grad_(False)
        eos=model.generation_config.eos_token_id
        eos_ids=[eos] if isinstance(eos,int) else list(eos or [])
        pad=model.generation_config.pad_token_id
        pad=pad if pad is not None else eos_ids[0]
        records=[json.loads(s) for s in (pool/'articles.jsonl').read_text().splitlines()]
        prompts=np.memmap(pool/'prompts.int32.bin',dtype='<i4',mode='r').reshape(-1,config['prompt_length'])
        manifest={'schema_version':1,'status':'GENERATING_DATA_ONLY_NO_TRAINING','config':config,'smoke':smoke,
                  'teacher':str(model_path),'model_revision':pm['model_revision'],'model_weights_sha256':pm['model_weights_sha256'],
                  'prompt_pool':str(pool),'prompt_manifest_sha256':sha256(pool/'manifest.json'),'eos_token_ids':eos_ids,
                  'pad_token_id':pad,'source_sha256':hashes,'splits':{},
                  'alignment':'Record j stores raw teacher logits from prompt + completion[:j], predicting completion[j]. j=0 is full-prefill output; its loss mask is 0.',
                  'sparse_distribution':'Top8 raw FP32 logits + int32 IDs; full-vocab FP32 logsumexp at T=1,2; chosen token raw FP32 logit even outside Top8. Not full-vocabulary logits.',
                  'mask':'1 only for accepted response positions 1..length-1, EOS included; prompt, first output, padding and rejected answers are 0.',
                  'sampling':'Exact full-vocabulary nucleus after temperature scaling. Stored logits are before all sampling transformations.',
                  'response_policy':'Three independently sampled attempts per prompt. Keep 2 or 3 distinct responses; reject entire prompt if fewer than 2. Global exact completion deduplication.',
                  'limitations':'English WikiText continuation by the original Base teacher; no instruction quality guarantee or state targets. Separate dataset, not wired into current training.'}
        seen=set(); batch_number=0
        with torch.inference_mode():
            for split,target in config['target_effective_tokens'].items():
                directory=output/split;directory.mkdir()
                candidates=[r['prompt_id'] for r in records if r['split']==split]
                count=0; items=[]; generated=0; accepted_count=0
                for offset in range(0,len(candidates),config['prompts_per_batch']):
                    selected=candidates[offset:offset+config['prompts_per_batch']]
                    pids=np.repeat(selected,config['answers_per_prompt']).astype('<i4')
                    batch=len(pids);width=config['max_new_tokens']
                    prompt_array=np.array(prompts[pids],dtype='<i4')
                    inputs=torch.from_numpy(prompt_array.astype('int64')).cuda()
                    cache=DynamicCache(config=model.config)
                    token_store=torch.full((batch,width),pad,dtype=torch.int32,device='cuda')
                    id_store=torch.zeros((batch,width,8),dtype=torch.int32,device='cuda')
                    logit_store=torch.zeros((batch,width,8),dtype=torch.float32,device='cuda')
                    norm_store=torch.zeros((batch,width,2),dtype=torch.float32,device='cuda')
                    selected_store=torch.zeros((batch,width),dtype=torch.float32,device='cuda')
                    lengths=torch.zeros(batch,dtype=torch.int32,device='cuda');done=torch.zeros(batch,dtype=torch.bool,device='cuda')
                    seed=config['seed']+batch_number
                    rng=torch.Generator(device='cuda').manual_seed(seed)
                    for position in range(width):
                        hidden=model.model.language_model(input_ids=inputs,past_key_values=cache,use_cache=True).last_hidden_state
                        raw=model.lm_head(hidden[:,-1,:])
                        token,ids,values,norm,sampled=sparse_and_sample(raw,config['sampling_temperature'],config['sampling_top_p'],rng)
                        active=~done
                        token=torch.where(active,token,pad)
                        token_store[:,position]=token.to(torch.int32)
                        id_store[:,position]=ids.to(torch.int32)
                        logit_store[:,position]=values;norm_store[:,position]=norm;selected_store[:,position]=sampled
                        lengths+=active.to(torch.int32)
                        for eid in eos_ids: done|=token==eid
                        inputs=token[:,None]
                        if bool(done.all()): break
                    a={'prompt_tokens':prompt_array,'prompt_ids':pids,'answer_index':np.tile(np.arange(3,dtype='<i4'),len(selected)),
                       'tokens':token_store.cpu().numpy(),'lengths':lengths.cpu().numpy(),'top8_ids':id_store.cpu().numpy(),
                       'top8_logits':logit_store.cpu().numpy(),'logsumexp':norm_store.cpu().numpy(),'sampled_logit':selected_store.cpu().numpy(),
                       'batch_seed':np.asarray(seed,dtype='<i8')}
                    a['accepted']=accept_responses(a['tokens'],a['lengths'],3,seen)
                    a['loss_mask']=supervision_mask(a['lengths'],a['accepted'],width)
                    effective=int(a['loss_mask'].sum())
                    filename=directory/f'batch-{batch_number:06d}.npz'
                    guard(filename, sum(v.nbytes for v in a.values()) + 65536)
                    with filename.with_suffix('.partial').open('wb') as f: np.savez(f,**a)
                    filename.with_suffix('.partial').replace(filename)
                    items.append({'file':str(filename.relative_to(output)),'bytes':filename.stat().st_size,'sha256':sha256(filename),
                                  'effective_targets':effective,'batch_seed':seed,'prompt_ids':selected})
                    count+=effective;generated+=int(a['lengths'].sum());accepted_count+=int(a['accepted'].sum());batch_number+=1
                    guard(output/'generation.jsonl', 4096)
                    with (output/'generation.jsonl').open('a') as log:
                        log.write(json.dumps({'split':split,'batch':batch_number,'effective_targets':count,'last_batch_targets':effective})+'\n')
                    del cache,hidden,raw,inputs,token_store,id_store,logit_store,norm_store,selected_store
                    if count>=target: break
                if count<target: raise RuntimeError(f'Prompt pool exhausted for {split}: {count} < {target}')
                manifest['splits'][split]={'effective_targets':count,'stored_generated_tokens':generated,'accepted_responses':accepted_count,'shards':items}
            assert all(not p.requires_grad and p.grad is None for p in model.parameters())
            assert sha256(weights[0])==manifest['model_weights_sha256']
            assert all(sha256(ROOT/name)==digest for name,digest in hashes.items())
            manifest['status']='GENERATED_PENDING_AUDIT_NO_TRAINING'
            write_json(output/'manifest.json',manifest)
            # Readback and teacher replay only after all requested data have finished.
            report=audit(output)
            replay(model,output,max_steps=32 if smoke else 16)
            torch.cuda.synchronize()
            manifest.update(status='COMPLETE_DATA_ONLY_NO_TRAINING',elapsed_seconds=time.monotonic()-started,
                            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                            total_effective_targets=sum(v['effective_targets'] for v in manifest['splits'].values()),
                            total_bytes=sum(i['bytes'] for v in manifest['splits'].values() for i in v['shards']),
                            audits={'readback':'PASS','teacher_prefix_replay':'PASS','base_weights_unchanged':'PASS','all_parameters_frozen':'PASS'})
            write_json(output/'manifest.json',manifest)
        print(json.dumps({'output':str(output),**{k:manifest[k] for k in ('status','elapsed_seconds','peak_allocated_mib','total_effective_targets','total_bytes','audits')},'splits':report['splits']},ensure_ascii=False,indent=2),flush=True)
    except BaseException:
        write_json(output/'failure.json',{'status':'FAILED','traceback':traceback.format_exc(),'elapsed_seconds':time.monotonic()-started})
        raise


def main():
    global STORAGE_ROOT, MAXIMUM_BYTES, RESERVE_BYTES
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/qwen35_08b_teacher_top8.json')
    p.add_argument('--prompt-pool',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--storage-root',type=Path,help='Shared budget root for prompt pool, smoke and production data')
    p.add_argument('--smoke',action='store_true')
    actions=p.add_mutually_exclusive_group()
    actions.add_argument('--prepare-prompts',action='store_true');actions.add_argument('--run',action='store_true');actions.add_argument('--audit',type=Path)
    args=p.parse_args();config=json.loads(args.config.read_text())
    assert config['stored_topk']==8 and config['answers_per_prompt']==3 and config['prompt_length']==256
    assert config['normalizer_temperatures']==[1.0,2.0]
    assert config['sampling_temperature']>0 and 0<config['sampling_top_p']<=1
    MAXIMUM_BYTES=min(config['maximum_artifact_bytes'],2_000_000_000)
    RESERVE_BYTES=max(config['minimum_free_bytes'],20_000_000_000)
    if args.storage_root:
        STORAGE_ROOT=args.storage_root.resolve()
        STORAGE_ROOT.mkdir(parents=True,exist_ok=True)
        if args.output and not args.output.resolve().is_relative_to(STORAGE_ROOT): p.error('--output must lie within --storage-root')
        check_storage(STORAGE_ROOT,1_000_000,MAXIMUM_BYTES,RESERVE_BYTES)
    elif args.run or args.prepare_prompts or args.audit:
        p.error('Writing requires --storage-root to enforce the shared 2 GB budget')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
    if args.prepare_prompts:
        if args.output is None: p.error('--prepare-prompts requires --output')
        prepare_prompts(config,args.output.resolve())
    elif args.run:
        if args.output is None or args.prompt_pool is None: p.error('--run requires --output and --prompt-pool')
        generate(config,args.prompt_pool.resolve(),args.output.resolve(),args.smoke)
    elif args.audit:
        print(json.dumps(audit(args.audit.resolve()),indent=2))
    else:
        print(json.dumps({'status':'PLAN_ONLY_NO_GENERATION','config':config,'training':False},indent=2))

if __name__=='__main__': main()
