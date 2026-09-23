"""Sparse original-teacher distributions. No training or state-alignment targets."""
from __future__ import annotations
import hashlib
import numpy as np


def article_split(key):
    bucket = int(hashlib.sha256(('teacher-top8-v1:' + key).encode()).hexdigest()[:16], 16) % 1000
    for end, split in ((880, 'train'), (920, 'gate'), (940, 'dev'), (970, 'calibration'), (1000, 'test')):
        if bucket < end:
            return split


def sparse_and_sample(logits, temperature, top_p, generator):
    """Unprocessed logits are saved; only sampling uses temperature and nucleus."""
    import torch
    raw = logits.float()
    ordered, indices = raw.sort(dim=-1, descending=True)
    top_values, top_ids = ordered[:, :8], indices[:, :8]
    normalizers = torch.stack([torch.logsumexp(raw / t, dim=-1) for t in (1., 2.)], dim=-1)
    probabilities = (ordered / temperature).softmax(-1)
    # Keep the first item crossing top_p (and always at least the highest item).
    removed = probabilities.cumsum(-1) - probabilities >= top_p
    weights = probabilities.masked_fill(removed, 0)
    sampled_rank = torch.multinomial(weights, 1, generator=generator)
    token = indices.gather(-1, sampled_rank).squeeze(-1)
    sampled_logit = raw.gather(-1, token[:, None]).squeeze(-1)
    return token, top_ids, top_values, normalizers, sampled_logit


def accept_responses(tokens, lengths, group_size, seen):
    """Keep 2-3 distinct nonempty responses per prompt; never count duplicates."""
    accepted = np.zeros(len(lengths), dtype=np.bool_)
    for start in range(0, len(lengths), group_size):
        candidates, local = [], set()
        for row in range(start, start + group_size):
            if lengths[row] < 2:  # token 1 is full-prefill and has zero training mask.
                continue
            digest = hashlib.sha256(tokens[row, :lengths[row]].astype('<i4').tobytes()).hexdigest()
            if digest not in seen and digest not in local:
                candidates.append((row, digest)); local.add(digest)
        if len(candidates) >= 2:
            for row, digest in candidates:
                accepted[row] = True; seen.add(digest)
    return accepted


def supervision_mask(lengths, accepted, width):
    positions = np.arange(width)[None, :]
    return ((positions > 0) & (positions < lengths[:, None]) & accepted[:, None]).astype('uint8')


def validate_shard(a, vocab=248320, eos_ids=()):
    """Independent CPU readback of alignment, probability mass, lengths and masks."""
    tokens, lengths = a['tokens'], a['lengths']
    n, width = tokens.shape
    active = np.arange(width)[None, :] < lengths[:, None]
    if not (np.all(lengths >= 1) and np.all(lengths <= width)):
        raise AssertionError('Invalid response lengths')
    assert a['top8_ids'].shape == (n, width, 8)
    assert a['top8_logits'].shape == (n, width, 8)
    assert a['logsumexp'].shape == (n, width, 2)
    assert a['sampled_logit'].shape == (n, width)
    assert np.array_equal(a['loss_mask'], supervision_mask(lengths, a['accepted'], width))
    ids, values, norm = a['top8_ids'][active], a['top8_logits'][active], a['logsumexp'][active]
    sampled, generated = a['sampled_logit'][active], tokens[active]
    assert np.all((ids >= 0) & (ids < vocab)) and np.all((generated >= 0) & (generated < vocab))
    assert np.all(np.diff(values, axis=-1) <= 0)
    assert np.all(np.diff(np.sort(ids, axis=-1), axis=-1) > 0)
    assert np.isfinite(values).all() and np.isfinite(norm).all() and np.isfinite(sampled).all()
    masses = []
    for col, temperature in enumerate((1., 2.)):
        mass = np.exp(values.astype('float64') / temperature - norm[:, col, None]).sum(-1)
        assert np.all(mass <= 1.00002) and np.all(mass > 0)
        assert np.all(sampled / temperature <= norm[:, col] + 2e-5)
        masses.append(mass)
    matched = ids == generated[:, None]
    if matched.any():
        assert np.allclose(values[matched], np.broadcast_to(sampled[:, None], values.shape)[matched], atol=0, rtol=0)
    for row, length in enumerate(lengths):
        if eos_ids:
            assert not np.isin(tokens[row, :length-1], eos_ids).any()
            if length < width:
                assert int(tokens[row, length-1]) in eos_ids
    for prompt in set(a['prompt_ids'].tolist()):
        rows = a['prompt_ids'] == prompt
        assert int(a['accepted'][rows].sum()) in (0, 2, 3)
    return {'stored_generated_tokens':int(lengths.sum()), 'effective_targets':int(a['loss_mask'].sum()),
            'accepted_responses':int(a['accepted'].sum()), 'top8_mass_t1_sum':float(masses[0].sum()),
            'top8_mass_t1_min':float(masses[0].min()), 'top8_mass_t1_max':float(masses[0].max())}


def check_storage(root, pending_bytes, maximum_bytes=2_000_000_000, reserve_bytes=20_000_000_000):
    """Bound all run artifacts together and leave an explicit free-space reserve."""
    from pathlib import Path
    import shutil
    root = Path(root)
    if pending_bytes < 0:
        raise ValueError('Negative storage reservation')
    used = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    if used + pending_bytes > maximum_bytes:
        raise RuntimeError(f'Artifact storage limit: {used + pending_bytes} > {maximum_bytes} bytes')
    if shutil.disk_usage(root).free - pending_bytes < reserve_bytes:
        raise RuntimeError('Free-space reserve would be violated; stop before writing')
    return used
