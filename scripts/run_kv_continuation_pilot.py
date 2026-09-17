#!/usr/bin/env python3
"""Small frozen KV-injection experiment. Writes results only on completion."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phase0.cache_tools import clone_dynamic_cache, assert_cache_storage_independent
from phase1.io import TokenSequenceStore, checkpoint_metadata, write_json
from phase1.models import build_trainable_probes
from phase1.teacher import require_corrected_checkpoint


def to_cache(attn, rotary, hidden, raw_k, raw_v):
    b, n, _ = hidden.shape
    key = attn.k_norm(raw_k.reshape(b, n, -1, attn.head_dim)).transpose(1, 2)
    pos = torch.arange(n, device=hidden.device)[None].expand(b, -1)
    cos, sin = rotary(hidden, pos)
    _, key = apply_rotary_pos_emb(key, key, cos, sin)
    value = raw_v.reshape(b, n, -1, attn.head_dim).transpose(1, 2)
    return key.contiguous(), value.contiguous()


def paired_ci(base, candidate, seed, draws):
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(base), (draws, len(base)))
    improvements = 1 - candidate[ix].mean(1) / np.maximum(base[ix].mean(1), 1e-12)
    return [float(x) for x in np.quantile(improvements, [0.025, 0.975])]


def summarize(records, config):
    summary = {}
    gates = config['gate']
    for length in config['context_lengths']:
        rows = [r for r in records if r['context_length'] == length]
        arms = {}
        for name in rows[0]['arms']:
            kl = np.array([r['arms'][name]['kl'] for r in rows])
            nll = np.array([r['arms'][name]['nll_delta'] for r in rows])
            top = np.array([r['arms'][name]['top1_equal'] for r in rows])
            arms[name] = {
                'mean_kl': float(kl.mean()), 'mean_nll_increase': float(nll.mean()),
                'top1_agreement': float(top.mean()),
                'early_1_8_kl': float(kl[:, :8].mean()),
                'late_97_128_kl': float(kl[:, 96:].mean()),
                'kl_by_step': kl.mean(0).tolist(),
                'checkpoints': {str(s): {'kl': float(kl[:, s-1].mean()), 'top1_agreement': float(top[:, s-1].mean())} for s in [1, 8, 32, 128]},
            }
        candidate = np.array([np.mean(r['arms']['predicted_all']['kl']) for r in rows])
        comparisons = {}
        for baseline in ['zero_all', 'original_all']:
            base = np.array([np.mean(r['arms'][baseline]['kl']) for r in rows])
            comparisons[baseline] = {'relative_improvement': float(1-candidate.mean()/max(base.mean(), 1e-12)),
                'paired_sequence_bootstrap_ci95': paired_ci(base, candidate, config['seed'], gates['bootstrap_samples'])}
        a = arms['predicted_all']
        checks = {
            'beats_both_baselines': all(c['relative_improvement'] >= gates['minimum_baseline_relative_improvement'] and c['paired_sequence_bootstrap_ci95'][0] > 0 for c in comparisons.values()),
            'mean_kl_acceptable': a['mean_kl'] <= gates['maximum_mean_kl'],
            'boundary_kl_acceptable': a['checkpoints']['1']['kl'] <= gates['maximum_mean_kl'],
            'nll_increase_acceptable': a['mean_nll_increase'] <= gates['maximum_mean_nll_increase'],
            'no_large_late_growth': a['late_97_128_kl'] <= gates['maximum_late_to_early_kl_ratio'] * a['early_1_8_kl'] + gates['growth_absolute_slack'],
        }
        summary[str(length)] = {'arms': arms, 'comparisons': comparisons, 'checks': checks,
            'decision': 'GO_TO_GDN_PILOT' if all(checks.values()) else 'HOLD_REPAIR_KV'}
    return summary


@torch.inference_mode()
def run(config, output):
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; no CPU fallback or fabricated measurements')
    torch.set_num_threads(4)
    start = time.perf_counter()
    model_path = ROOT / 'models/Qwen3.5-0.8B-Base'
    model = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa').eval().cuda()
    lm = model.model.language_model
    cfg = model.config.text_config
    layers = [15, 19, 23]
    probes, hashes = {}, {}
    for layer in layers:
        path = ROOT / config.get('checkpoint_dir', 'checkpoints/phase1') / f'layer{layer}_kv_probes.pt'
        hashes[str(layer)] = hashlib.sha256(path.read_bytes()).hexdigest()
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if config.get('require_corrected_checkpoint'):
            require_corrected_checkpoint(saved)
        group = build_trainable_probes(cfg.hidden_size, cfg.num_key_value_heads*cfg.head_dim, [3,7,11], [64,128,256])
        group.load_state_dict(saved['probe_state_dict'])
        probes[layer] = group[config['selected_methods'][str(layer)]].cuda().eval()
    store = TokenSequenceStore(ROOT / 'data/phase1_wikitext103', 'test')
    flat = store.tokens.reshape(-1)
    # Disjoint windows across both lengths, distributed across the packed test stream.
    stride = max(config['context_lengths']) + config['decode_steps']
    slots = np.arange(len(flat)//stride)
    if config.get('exclude_windows_from'):
        previous = json.loads((ROOT / config['exclude_windows_from']).read_text())
        excluded = [(r['test_token_offset'], r['test_token_offset']+r['context_length']+previous['config']['decode_steps']) for r in previous['records']]
        slots = np.array([s for s in slots if all(s*stride+stride <= a or s*stride >= b for a,b in excluded)])
    rng = np.random.default_rng(config['seed'])
    rng.shuffle(slots)
    assert len(slots) >= len(config['context_lengths'])*config['samples_per_length']
    records, validations = [], []
    arm_names = ['oracle_reencoded', 'predicted_15', 'predicted_19', 'predicted_23', 'predicted_all', 'original_all', 'zero_all']
    sample_slot = 0
    for length in config['context_lengths']:
        for batch_start in range(0, config['samples_per_length'], config['batch_size']):
            count = min(config['batch_size'], config['samples_per_length']-batch_start)
            offsets = (slots[sample_slot:sample_slot+count]*stride).tolist()
            sample_slot += count
            tokens = torch.tensor(np.stack([flat[o:o+length+config['decode_steps']] for o in offsets]).astype(np.int64), device='cuda')
            sources, target_inputs, handles = {}, {}, []
            def source_hook(i):
                def hook(m, args, out):
                    sources[i] = (out[0] if isinstance(out, tuple) else out).detach()
                return hook
            def target_hook(i):
                def hook(m, args, kwargs):
                    target_inputs[i] = (args[0] if args else kwargs['hidden_states']).detach()
                return hook
            for i in [3,7,11]:
                handles.append(lm.layers[i].register_forward_hook(source_hook(i)))
            for i in layers:
                # Actual self-attention input includes input_layernorm.
                handles.append(lm.layers[i].self_attn.register_forward_pre_hook(target_hook(i), with_kwargs=True))
            try:
                prefix = lm(input_ids=tokens[:, :length-1], use_cache=True).past_key_values
            finally:
                for h in handles:
                    h.remove()
            caches = {name: clone_dynamic_cache(prefix, cfg) for name in arm_names}
            check = {'context_length': length, 'offsets': offsets, 'oracle_cache_max_abs': 0.0}
            for cache in caches.values():
                assert_cache_storage_independent(prefix, cache)
            for layer in layers:
                attn = lm.layers[layer].self_attn
                hidden = target_inputs[layer]
                exact = to_cache(attn, lm.rotary_emb, hidden, attn.k_proj(hidden), attn.v_proj(hidden))
                for a, b in zip(exact, (prefix.layers[layer].keys, prefix.layers[layer].values)):
                    check['oracle_cache_max_abs'] = max(check['oracle_cache_max_abs'], (a-b).abs().max().item())
                    if not torch.equal(a, b):
                        raise AssertionError(f'Oracle KV reencoding mismatch layer {layer}')
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    k, v = probes[layer](sources)
                predicted = to_cache(attn, lm.rotary_emb, sources[11], k, v)
                original_input = lm.layers[layer].input_layernorm(sources[11]) if config.get('normalized_original_projection') else sources[11]
                original = to_cache(attn, lm.rotary_emb, sources[11], attn.k_proj(original_input), attn.v_proj(original_input))
                updates = {'oracle_reencoded': exact, f'predicted_{layer}': predicted,
                    'predicted_all': predicted, 'original_all': original,
                    'zero_all': tuple(torch.zeros_like(x) for x in exact)}
                for arm, pair in updates.items():
                    caches[arm].layers[layer].keys = pair[0].clone()
                    caches[arm].layers[layer].values = pair[1].clone()
            batch_records = [{'context_length': length, 'test_token_offset': o, 'arms': {
                name: {'kl': [], 'nll_delta': [], 'top1_equal': []} for name in arm_names}} for o in offsets]
            # Step 1 consumes the final prompt token and predicts the first continuation token.
            # Every branch then consumes the same held-out token at every subsequent step.
            for step in range(config['decode_steps']):
                token = tokens[:, length-1+step:length+step]
                target = tokens[:, length+step]
                teacher = lm(input_ids=token, past_key_values=prefix, use_cache=True)
                reference_logits = model.lm_head(teacher.last_hidden_state[:, -1]).float()
                reference_logp = F.log_softmax(reference_logits, -1)
                reference_prob = reference_logp.exp()
                teacher_nll = -reference_logp.gather(1, target[:, None]).squeeze(1)
                for arm, cache in caches.items():
                    student = lm(input_ids=token, past_key_values=cache, use_cache=True)
                    logits = model.lm_head(student.last_hidden_state[:, -1]).float()
                    if arm == 'oracle_reencoded' and not torch.equal(reference_logits, logits):
                        raise AssertionError('Oracle continuation differs from teacher')
                    logp = F.log_softmax(logits, -1)
                    kl = (reference_prob*(reference_logp-logp)).sum(-1)
                    delta = -logp.gather(1, target[:, None]).squeeze(1)-teacher_nll
                    agreement = logits.argmax(-1).eq(reference_logits.argmax(-1))
                    if not torch.isfinite(kl).all() or not torch.isfinite(delta).all():
                        raise AssertionError(f'Non-finite metrics: {arm}')
                    for row, k, d, a in zip(batch_records, kl.tolist(), delta.tolist(), agreement.tolist()):
                        row['arms'][arm]['kl'].append(k)
                        row['arms'][arm]['nll_delta'].append(d)
                        row['arms'][arm]['top1_equal'].append(a)
            validations.append(check)
            records.extend(batch_records)
            del caches, prefix, sources, target_inputs
    summary = summarize(records, config)
    result = {'config': config, 'checkpoint_sha256': hashes,
        'environment': {'torch': torch.__version__, 'transformers': transformers.__version__,
            'gpu': torch.cuda.get_device_name(), 'dtype': 'bfloat16', 'attention_backend': 'sdpa',
            **checkpoint_metadata(model_path)},
        'elapsed_seconds': time.perf_counter()-start, 'validations': validations,
        'summary': summary, 'records': records,
        'decision': 'GO_TO_GDN_PILOT' if all(s['decision']=='GO_TO_GDN_PILOT' for s in summary.values()) else 'HOLD_REPAIR_KV'}
    write_json(output/'results.json', result)
    lines = ['# K/V 缓存注入与续写小规模实验', '', f"结论：**{result['decision']}**", '',
        '固定已有 checkpoint；256/2048 长度各 16 个测试窗口；每个窗口 128 步相同 token 续写。保留 teacher GDN 状态。',
        'Step 1 为最后一个 prompt token 经过缓存路径后预测首个续写 token。测试窗口互不重叠，但来自 Phase 1 已用过的 test split；并非新的独立语料。',
        '门槛在运行前固定，是本轮工程筛选标准，不代表普适质量阈值。未测试自由生成、GDN 重建或实际加速。', '',
        '| Context | Method | Mean KL | ΔNLL (nats/token) | Top-1 agreement | KL@1 | KL@32 | KL@128 |',
        '|---:|:---|---:|---:|---:|---:|---:|---:|']
    for length, s in summary.items():
        for arm, a in s['arms'].items():
            lines.append(f"| {length} | {arm} | {a['mean_kl']:.6f} | {a['mean_nll_increase']:.6f} | {a['top1_agreement']:.2%} | {a['checkpoints']['1']['kl']:.6f} | {a['checkpoints']['32']['kl']:.6f} | {a['checkpoints']['128']['kl']:.6f} |")
    for length, s in summary.items():
        lines += ['', f"长度 {length}：{s['decision']}；检查：`{json.dumps(s['checks'], ensure_ascii=False)}`", '',
            '基线比较（以窗口为单位配对 bootstrap，不能代表多随机种子训练置信度）：',
            '```json', json.dumps(s['comparisons'], ensure_ascii=False, indent=2), '```', '']
    lines += ['', '所有 batch 均验证 teacher K/V 经 KNorm/RoPE 重建逐位相等，oracle 128 步 logits 与 teacher 逐位相等。',
        f"运行耗时：{result['elapsed_seconds']:.1f} 秒。", '',
        '决策规则：两种长度均通过，才支持进入单层 GDN pilot；否则先修复 K/V 或收缩替换范围，不据此否定所有 CED 方案。']
    (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'decision': result['decision'], 'elapsed_seconds': result['elapsed_seconds'], 'report': str(output/'report.md')}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(ROOT/'configs/kv_continuation_pilot.json'))
    parser.add_argument('--output-dir', default=str(ROOT/'results/kv_continuation_pilot'))
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    run(json.loads(Path(args.config).read_text()), out)
