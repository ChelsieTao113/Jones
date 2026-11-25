"""Independent three-polarization, shared-Jones Fourier GS benchmark.

All arrays use row=y, column=x; Jones indices are [output, input].
The forward FFT is fftshift(fft2(field, axes=(0,1), norm='ortho'));
the inverse is ifft2(ifftshift(field), axes=(0,1), norm='ortho').
No analyzer is present. This module deliberately does not import legacy AP code.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_ID = 'symmetric_unitary_total_intensity_fourier_v1'
SOLVER_ID = 'multichannel_vector_gs_v1'
CANONICAL = {
    'X': [1, 0], 'Y': [0, 1],
    'D': [1 / math.sqrt(2), 1 / math.sqrt(2)],
    'A': [1 / math.sqrt(2), -1 / math.sqrt(2)],
    'C+': [1 / math.sqrt(2), 1j / math.sqrt(2)],
    'C-': [1 / math.sqrt(2), -1j / math.sqrt(2)],
}


def jones_from_angles(phi1, phi2, theta):
    a, b = np.exp(1j * phi1), np.exp(1j * phi2)
    c, s = np.cos(theta), np.sin(theta)
    out = np.empty(phi1.shape + (2, 2), dtype=np.result_type(a, b))
    out[..., 0, 0] = c*c*a + s*s*b
    out[..., 0, 1] = c*s*(b-a)  # R(-theta) diag(a,b) R(theta)
    out[..., 1, 0] = out[..., 0, 1]
    out[..., 1, 1] = s*s*a + c*c*b
    return out


def random_jones(n, seed, dtype=np.complex64):
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0, 2*np.pi, (2, n, n))
    theta = rng.uniform(0, np.pi, (n, n))
    return jones_from_angles(phi[0], phi[1], theta).astype(dtype)


def canonicalize_p(p):
    p = np.asarray(p, dtype=np.complex128)
    p /= np.linalg.norm(p)
    first = next((z for z in p if abs(z) > 1e-12), None)
    if first is None:
        raise ValueError('zero polarization')
    return (p * np.exp(-1j*np.angle(first))).astype(np.complex64)


def sample_polarizations(seed, bucket='continuous'):
    rng = np.random.default_rng(seed)
    keys = list(CANONICAL)
    if bucket == 'canonical':
        p = [canonicalize_p(CANONICAL[k]) for k in rng.choice(keys, 3, replace=False)]
    else:
        how_many = 2 if bucket == 'mixed' else 3
        p = []
        if bucket == 'mixed':
            p.append(canonicalize_p(CANONICAL[str(rng.choice(keys))]))
        for _ in range(how_many):
            u, delta = rng.uniform(-1, 1), rng.uniform(0, 2*np.pi)
            alpha = 0.5 * np.arccos(u)
            p.append(canonicalize_p([np.cos(alpha), np.exp(1j*delta)*np.sin(alpha)]))
    p = np.stack(p)
    rng.shuffle(p, axis=0)
    fidelity = np.abs(p.conj() @ p.T)**2
    np.fill_diagonal(fidelity, 0)
    cond = float(np.linalg.cond(p.T @ p.conj()))
    return p, float(fidelity.max()), cond


def forward(j, p):
    """Return complex U[K,H,W,2] and total intensity I[K,H,W]."""
    exit_field = np.einsum('yxoi,ki->kyxo', j, p, optimize=True)
    u = np.fft.fftshift(np.fft.fft2(exit_field, axes=(1, 2), norm='ortho'), axes=(1, 2))
    return u, np.sum(np.abs(u)**2, axis=-1).real


def inverse(u):
    return np.fft.ifft2(np.fft.ifftshift(u, axes=(1, 2)), axes=(1, 2), norm='ortho')


def roi_slices(n):
    m = n//2
    start = (n-m)//2
    return slice(start, start+m), slice(start, start+m)


def project_intensity(u, target, seed):
    n = u.shape[1]
    ys, xs = roi_slices(n)
    out = u.copy()
    local = out[:, ys, xs, :]
    power = np.sum(np.abs(local)**2, axis=-1)
    amplitude = np.sqrt(target)
    good = power > 1e-20
    out_local = np.zeros_like(local)
    np.divide(local * amplitude[..., None], np.sqrt(power)[..., None],
              out=out_local, where=good[..., None])
    missing = (~good) & (target > 0)
    if missing.any():
        rng = np.random.default_rng(seed)
        z = rng.normal(size=local.shape) + 1j*rng.normal(size=local.shape)
        z /= np.linalg.norm(z, axis=-1, keepdims=True)
        out_local[missing] = (z * amplitude[..., None])[missing]
    out[:, ys, xs, :] = out_local
    return out


def takagi_project(z):
    """Closest symmetric unitary, via the real symmetric Takagi embedding.

    If S=A+iB, eigenvectors of [[A,B],[B,-A]] with nonnegative eigenvalues
    yield Takagi vectors v=x+iy satisfying S conjugate(v)=sigma*v.
    """
    s = 0.5*(z + np.swapaxes(z, -1, -2))
    a, b = s.real, s.imag
    k = np.concatenate((np.concatenate((a, b), axis=-1),
                        np.concatenate((b, -a), axis=-1)), axis=-2)
    values, vectors = np.linalg.eigh(k)
    v = vectors[..., :2, 2:] + 1j*vectors[..., 2:, 2:]
    # At exactly zero, the Takagi basis is nonunique; choose identity.
    zero = np.max(np.abs(s), axis=(-2, -1)) < 1e-14
    if np.any(zero):
        v[zero] = np.eye(2, dtype=np.complex128)
    j = v @ np.swapaxes(v, -1, -2)
    return j.astype(z.dtype, copy=False), values[..., 2:], v


def target_pattern(n, seed, kind=None):
    """Antialiased procedural shape, with a 6-pixel margin in a 64 ROI."""
    rng = np.random.default_rng(seed)
    kind = kind or str(rng.choice(['disk', 'ring', 'rectangle', 'cross', 'bars', 'triangle']))
    m = n//2
    scale = 3
    yy, xx = np.mgrid[:m*scale, :m*scale]
    shift_x, shift_y = rng.uniform(-4, 4, size=2)
    x = (xx+0.5)/scale - m/2 - shift_x
    y = (yy+0.5)/scale - m/2 - shift_y
    r = np.hypot(x, y)
    rad = rng.uniform(10, max(11, m/2-9))
    parameters = dict(kind=kind, seed=seed, shift_x=float(shift_x),
                      shift_y=float(shift_y), radius=float(rad), supersampling=scale)
    if kind == 'disk':
        g = r < rad
    elif kind == 'ring':
        width = rng.uniform(4, 8)
        g = (r < rad) & (r > rad-width)
        parameters['width'] = float(width)
    elif kind == 'rectangle':
        ratio = rng.uniform(.4, .9)
        g = (np.abs(x) < rad) & (np.abs(y) < rad*ratio)
        parameters['ratio'] = float(ratio)
    elif kind == 'cross':
        w = rng.uniform(3, 6)
        g = ((np.abs(x) < w) & (np.abs(y) < rad)) | ((np.abs(y) < w) & (np.abs(x) < rad))
        parameters['width'] = float(w)
    elif kind == 'bars':
        frequency = rng.uniform(.4, .7)
        g = (np.abs(y) < rad) & (np.abs(x) < rad) & (np.cos(x*frequency) > .25)
        parameters['frequency'] = float(frequency)
    elif kind == 'triangle':
        g = (y > -rad*.7) & (y < rad) & (np.abs(x) < (rad-y)*.6)
    else:
        raise ValueError(kind)
    g = g.reshape(m, scale, m, scale).mean(axis=(1, 3))
    g[:6] = 0; g[-6:] = 0; g[:, :6] = 0; g[:, -6:] = 0
    if not g.sum():
        raise ValueError('empty target')
    return g.astype(np.float32), parameters


def exact_zero_conflicts(target, p):
    """Count pixels where two independent inputs are zero but a third is not."""
    conflicts = 0
    for a in range(3):
        for b in range(a+1, 3):
            if abs(np.linalg.det(p[[a, b]])) < 1e-5:
                continue
            c = 3-a-b
            conflicts += int(np.count_nonzero((target[a] == 0) & (target[b] == 0) & (target[c] > 0)))
    return conflicts


def target_from_shape(shapes, n, beta):
    shapes = np.asarray(shapes, dtype=np.float64)
    if not np.isfinite(shapes).all() or np.min(shapes) < 0 or np.any(shapes.sum((1, 2)) <= 0):
        raise ValueError('invalid target shape')
    t = shapes / shapes.sum((1, 2), keepdims=True)
    return t.astype(np.float32), (t * (beta*n*n)).astype(np.float32)


def corr(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    aa, bb = a-a.mean(), b-b.mean()
    d = np.linalg.norm(aa)*np.linalg.norm(bb)
    return None if d < 1e-20 else float(np.dot(aa, bb)/d)


def metrics(intensity, target, beta):
    n = intensity.shape[-1]
    ys, xs = roi_slices(n)
    observed = intensity[:, ys, xs]
    power = n*n
    out = []
    for channel, (i, t, this_beta) in enumerate(zip(observed, target, np.broadcast_to(beta, (len(target),)))):
        t64 = t.astype(np.float64)
        i64 = i.astype(np.float64)
        e_abs = float(np.sum((i64-t64)**2)/(np.sum(t64*t64)+1e-20))
        shape_i = i64/(i64.sum()+1e-20)
        shape_t = t64/(t64.sum()+1e-20)
        e_shape = float(np.sum((shape_i-shape_t)**2)/(np.sum(shape_t*shape_t)+1e-20))
        eta = float(i64.sum()/power)
        support = t64 >= .05*t64.max()
        black = ~support
        leakage = float(i64[black].sum()/(i64.sum()+1e-20)) if black.any() else None
        # Fixed physical scale: mean target intensity across the ROI is one.
        # This is shared by all tasks with the same beta/grid, never fitted to a sample.
        roi_mean = this_beta*power/t64.size
        x, y = i64/roi_mean, t64/roi_mean
        data_range = 16.0
        mse = float(np.mean((x-y)**2))
        ssim = ((2*x.mean()*y.mean()+(.01*data_range)**2)*
                (2*np.mean((x-x.mean())*(y-y.mean()))+(.03*data_range)**2) /
                ((x.mean()**2+y.mean()**2+(.01*data_range)**2)*
                 (x.var()+y.var()+(.03*data_range)**2)))
        out.append(dict(e_abs=e_abs, e_shape=e_shape, pcc=corr(x, y),
                        psnr=None if mse == 0 else float(10*np.log10(data_range**2/mse)),
                        ssim=float(ssim), eta_roi=eta,
                        eta_support=float(i64[support].sum()/power),
                        black_leakage=leakage,
                        outside_power=float((intensity[channel].sum()-i64.sum())/power),
                        eta_relative_error=float(abs(eta-this_beta)/this_beta)))
    return out


def physical_residuals(j, p, intensity):
    n = j.shape[0]
    jj = np.swapaxes(j.conj(), -1, -2) @ j
    identity = np.eye(2)
    return dict(symmetry=float(np.max(np.abs(j-np.swapaxes(j, -1, -2)))),
                unitarity=float(np.max(np.abs(jj-identity))),
                power=float(np.max(np.abs(intensity.sum((1,2))-n*n)/(n*n))),
                polarization=float(np.max(np.abs(np.sum(np.abs(p)**2, axis=-1)-1))))


def accepted(per_channel, residual, policy):
    hard = max(residual.values()) <= policy['physical_tolerance']
    return hard and all(m['pcc'] is not None and m['pcc'] >= policy['min_pcc']
                        and m['e_abs'] <= policy['max_e_abs']
                        and m['eta_relative_error'] <= policy['max_eta_relative_error']
                        and (m['black_leakage'] is None or m['black_leakage'] <= policy['max_black_leakage'])
                        for m in per_channel)


def evaluate(j, p, target, beta, policy):
    _, intensity = forward(j, p)
    m = metrics(intensity, target, beta)
    r = physical_residuals(j, p, intensity)
    score = (max(v['e_abs'] for v in m), float(np.mean([v['e_abs'] for v in m])))
    return dict(intensity=intensity, per_channel=m, residual=r,
                score=score, accepted=accepted(m, r, policy))


def solve(p, target, beta, n, seed, config, policy):
    started = time.monotonic()
    j = random_jones(n, seed)
    pinv = np.linalg.pinv(p.T.astype(np.complex128))
    best = None
    history = []
    last_improvement = 0
    best_mean = float('inf')
    for iteration in range(1, config['max_iterations']+1):
        u, _ = forward(j, p)
        q = inverse(project_intensity(u, target, seed))
        z = np.einsum('kyxo,ki->yxoi', q, pinv, optimize=True)
        j, _, _ = takagi_project(z)
        if iteration % config['evaluate_every'] and iteration != config['max_iterations']:
            continue
        e = evaluate(j, p, target, beta, policy)
        history.append(dict(iteration=iteration, max_e_abs=e['score'][0], mean_e_abs=e['score'][1],
                            accepted=e['accepted']))
        key = (0 if e['accepted'] else 1, *e['score'])
        if best is None or key < best[0]:
            best = (key, j.copy(), e, iteration)
        if e['score'][1] < best_mean*(1-config['min_relative_improvement']):
            best_mean = e['score'][1]
            last_improvement = iteration
        if e['accepted']:
            break
        if (iteration >= config['min_iterations'] and
                iteration-last_improvement >= config['patience_iterations']):
            break
    return dict(jones=best[1], evaluation=best[2], best_iteration=best[3],
                iterations=iteration, history=history, init_seed=seed,
                time_seconds=time.monotonic()-started)


def make_task(index, split, n, beta, root_seed, family='independent'):
    offsets = dict(train=0, val=100000000, test=200000000,
                   dev=300000000, demo=400000000, challenge=500000000)
    seed = root_seed + offsets[split] + index*1009
    # This deterministic map gives 30/50/20 percent by task index.
    if index % 10 in (3, 4, 5, 6, 7):
        bucket = 'continuous'
    elif index % 10 in (8, 9):
        bucket = 'mixed'
    else:
        bucket = 'canonical'
    p, fidelity, cond = sample_polarizations(seed, bucket)
    shapes, kinds, augmentations = [], [], []
    if family == 'constructed':
        true_j = random_jones(n, seed+31)
        _, full = forward(true_j, p)
        ys, xs = roi_slices(n)
        # Constructed task uses physically realized target power, not beta scaling.
        target = full[:, ys, xs].astype(np.float32)
        shape = target / target.sum((1,2), keepdims=True)
        beta_actual = (target.sum((1,2))/(n*n)).tolist()
        kinds = ['constructed']*3
        augmentations = [dict(seed=seed+31, generator='random_jones_v1')]*3
    else:
        for k in range(3):
            g, params = target_pattern(n, seed+51+k*17)
            shapes.append(g); kinds.append(params['kind']); augmentations.append(params)
        shape, target = target_from_shape(shapes, n, beta)
        beta_actual = [beta]*3
        true_j = None
    return dict(task_id='%s_%06d' % (split, index), split=split, seed=seed,
                family=family, polarization=p, target_shape=shape,
                target_intensity=target, beta=beta_actual, kinds=kinds,
                polarization_bucket=bucket,
                augmentations=augmentations,
                zero_conflicts=exact_zero_conflicts(target, p),
                image_ids=['%s_%d_%d' % (split, seed, k) for k in range(3)],
                fidelity=fidelity, gram_condition=cond, known_j=true_j)


def jsonable(x):
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, Path): return str(x)
    raise TypeError(type(x).__name__)


def save_npz(path, task, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.npz')
    e = result['evaluation']
    ys, xs = roi_slices(result['jones'].shape[0])
    np.savez_compressed(tmp, jones=result['jones'].astype(np.complex64),
                        polarization=task['polarization'], target_shape=task['target_shape'],
                        target_intensity=task['target_intensity'],
                        reconstruction_full=e['intensity'].astype(np.float32),
                        reconstruction_roi=e['intensity'][:,ys,xs].astype(np.float32),
                        beta=np.asarray(task['beta'], np.float32))
    with np.load(tmp, allow_pickle=False) as z:
        if set(z.files) != {'jones','polarization','target_shape','target_intensity','reconstruction_full','reconstruction_roi','beta'}:
            raise IOError('incomplete archive')
    os.replace(tmp, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def append_jsonl(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=jsonable)+'\n')
        f.flush(); os.fsync(f.fileno())


def run(args):
    cfg = json.loads((ROOT/'configs'/'runtime.json').read_text(encoding='utf-8'))
    policy = json.loads((ROOT/'configs'/'quality_policy.json').read_text(encoding='utf-8'))
    n = args.n or cfg['n']
    beta = args.beta or cfg['beta']
    if n < 32 or n % 2 or not (0 < beta <= 1) or args.tasks < 1 or args.seeds < 1 or args.walltime_hours <= 0 or args.constructed_every < 1:
        raise ValueError('invalid grid, beta, task count, seeds, or walltime')
    if args.split in ('train', 'val', 'test'):
        planned = cfg['planned_counts'][args.split+'_accepted']
        if args.accepted_target is None or not (1 <= args.accepted_target <= planned):
            raise ValueError('production requires --accepted-target up to %d' % planned)
    elif args.accepted_target is not None:
        raise ValueError('--accepted-target is production-only')
    validation = ROOT/'reports'/'validation.json'
    if not validation.exists() or json.loads(validation.read_text(encoding='utf-8')).get('status') != 'passed':
        raise RuntimeError('Stage A validation must pass before generation')
    if args.split == 'dev':
        demo = ROOT/'runs'/'demo_v1'/'manifest.jsonl'
        if not demo.exists() or len(demo.read_text(encoding='utf-8').splitlines()) < 10:
            raise RuntimeError('Complete all ten demo task records first')
    if args.split in ('train', 'val', 'test'):
        freeze = ROOT/'configs'/'production_freeze.json'
        if not freeze.exists():
            raise RuntimeError('Production gate closed: freeze after the 300-task pilot report')
        frozen = json.loads(freeze.read_text(encoding='utf-8'))
        if frozen.get('approved') is not True or frozen.get('runtime_sha256') != hashlib.sha256((ROOT/'configs'/'runtime.json').read_bytes()).hexdigest() or frozen.get('quality_sha256') != hashlib.sha256((ROOT/'configs'/'quality_policy.json').read_bytes()).hexdigest():
            raise RuntimeError('Production configuration is not frozen or hashes changed')
        if n != cfg['n'] or beta != cfg['beta']:
            raise RuntimeError('Production cannot override frozen n or beta')
    out = ROOT/'runs'/args.name
    out.mkdir(parents=True, exist_ok=True)
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    runtime_sha = hashlib.sha256((ROOT/'configs'/'runtime.json').read_bytes()).hexdigest()
    quality_sha = hashlib.sha256((ROOT/'configs'/'quality_policy.json').read_bytes()).hexdigest()
    manifest = out/'manifest.jsonl'
    prior = {}
    if manifest.exists():
        for line in manifest.read_text(encoding='utf-8').splitlines():
            row = json.loads(line); prior[row['task_id']] = row
        for row in prior.values():
            if 'path' in row:
                source = ROOT/row['path']
                if not source.exists() or hashlib.sha256(source.read_bytes()).hexdigest() != row['sha256']:
                    raise IOError('resume source missing or corrupt: '+row['task_id'])
    job_started = time.monotonic()
    accepted_count = sum(r.get('status') == 'accepted' for r in prior.values())
    for index in range(args.tasks):
        if args.accepted_target is not None and accepted_count >= args.accepted_target:
            break
        if time.monotonic()-job_started >= args.walltime_hours*3600:
            print('Walltime slice reached; manifest is resumable.', flush=True)
            break
        if (hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != source_sha or
            hashlib.sha256((ROOT/'configs'/'runtime.json').read_bytes()).hexdigest() != runtime_sha or
            hashlib.sha256((ROOT/'configs'/'quality_policy.json').read_bytes()).hexdigest() != quality_sha):
            raise RuntimeError('source/config changed during job; restart under a new run version')
        task = make_task(index, args.split, n, beta, cfg['seed'],
                         'constructed' if index % args.constructed_every == 0 else 'independent')
        if task['task_id'] in prior: continue
        if task['fidelity'] > cfg['max_pair_fidelity'] or task['gram_condition'] > cfg['max_gram_condition']:
            append_jsonl(manifest, dict(task_id=task['task_id'], status='invalid_condition',
                                        split=args.split, family=task['family'],
                                        polarization_bucket=task['polarization_bucket'],
                                        fidelity=task['fidelity'], gram_condition=task['gram_condition'],
                                        zero_conflicts=task['zero_conflicts']))
            continue
        best = None
        all_runs = []
        for attempt in range(args.seeds):
            r = solve(task['polarization'], task['target_intensity'], task['beta'], n,
                      task['seed']+100000+attempt*7919, cfg['solver'], policy)
            all_runs.append(dict(init_seed=r['init_seed'], iterations=r['iterations'],
                                 best_iteration=r['best_iteration'], score=r['evaluation']['score'],
                                 accepted=r['evaluation']['accepted'], time_seconds=r['time_seconds']))
            if best is None or (0 if r['evaluation']['accepted'] else 1, *r['evaluation']['score']) < (0 if best['evaluation']['accepted'] else 1, *best['evaluation']['score']):
                best = r
            if r['evaluation']['accepted']: break
        filename = out/'solutions'/(task['task_id']+'.npz')
        digest = save_npz(filename, task, best)
        e = best['evaluation']
        ys, xs = roi_slices(n)
        roi_actual = e['intensity'][:, ys, xs]
        crosstalk = [[corr(roi_actual[i], task['target_intensity'][j]) for j in range(3)] for i in range(3)]
        target_similarity = [[corr(task['target_intensity'][i], task['target_intensity'][j]) for j in range(3)] for i in range(3)]
        bright_fraction = [float(np.mean(g >= .05*g.max())) for g in task['target_shape']]
        state = ('accepted' if e['accepted'] else
                 'solver_not_converged' if e['score'][0] > policy['max_e_abs'] else
                 'quality_rejected')
        row = dict(task_id=task['task_id'], split=args.split, family=task['family'],
                   status=state,
                   model_id=MODEL_ID, solver_id=SOLVER_ID, policy_id=policy['policy_id'],
                   task_seed=task['seed'], init_seed=best['init_seed'],
                   kinds=task['kinds'], image_ids=task['image_ids'],
                   polarization_bucket=task['polarization_bucket'],
                   augmentation=task['augmentations'], zero_conflicts=task['zero_conflicts'],
                   bright_fraction=bright_fraction, crosstalk=crosstalk,
                   target_similarity=target_similarity,
                   fidelity=task['fidelity'], gram_condition=task['gram_condition'],
                   beta=task['beta'], per_channel=e['per_channel'], residual=e['residual'],
                   score=e['score'], best_iteration=best['best_iteration'],
                   iterations=best['iterations'], time_seconds=sum(x['time_seconds'] for x in all_runs),
                   solver_runs=len(all_runs), all_runs=all_runs,
                   aggregate=dict(mean_e_abs=e['score'][1], worst_e_abs=e['score'][0],
                                  worst_channel=int(np.argmax([m['e_abs'] for m in e['per_channel']]))),
                   metric_version=2,
                   provenance=dict(source_sha256=source_sha,
                                   runtime_sha256=runtime_sha,
                                   quality_sha256=quality_sha,
                                   python=platform.python_version(), numpy=np.__version__,
                                   machine=platform.platform()),
                   path=str(filename.relative_to(ROOT)), sha256=digest,
                   history=best['history'])
        append_jsonl(manifest, row)
        accepted_count += int(e['accepted'])
        print('%s %s max_e_abs=%.5g' % (row['task_id'], row['status'], e['score'][0]), flush=True)
    print('manifest:', manifest)


def validate():
    p = np.array([CANONICAL[k] for k in ('X','Y','D','C+')], np.complex128)
    n = 16
    rng = np.random.default_rng(51)
    j = random_jones(n, 41, np.complex128)
    assert np.max(np.abs(j-np.swapaxes(j,-1,-2))) < 1e-14
    assert np.max(np.abs(np.swapaxes(j.conj(),-1,-2)@j-np.eye(2))) < 1e-12
    assert np.max(np.abs(jones_from_angles(np.zeros((1,1)),np.zeros((1,1)),
        np.zeros((1,1)))[0,0]-np.eye(2))) < 1e-14
    half_wave = jones_from_angles(np.zeros((1,1)),np.full((1,1),np.pi),
                                  np.full((1,1),np.pi/4))[0,0]
    assert np.max(np.abs(np.abs(half_wave @ p[0])-np.array([0,1]))) < 1e-12
    quarter_wave = np.diag([1, 1j])
    assert np.max(np.abs(quarter_wave@p[0]-np.array([1,0]))) < 1e-12
    assert np.max(np.abs(quarter_wave@p[1]-np.array([0,1j]))) < 1e-12
    assert np.max(np.abs(quarter_wave@p[2]-np.array([1,1j])/np.sqrt(2))) < 1e-12
    assert np.max(np.abs(quarter_wave@p[3]-np.array([1,-1])/np.sqrt(2))) < 1e-12
    u, intensity = forward(j, p)
    assert np.max(np.abs(intensity.sum((1,2))-n*n)) < 1e-9
    assert np.max(np.abs(inverse(u)-np.einsum('yxoi,ki->kyxo',j,p))) < 1e-12
    exit_three = np.einsum('yxoi,ki->kyxo', j, p[:3])
    recovered = np.einsum('kyxo,ki->yxoi', exit_three, np.linalg.pinv(p[:3].T))
    assert np.max(np.abs(recovered-j)) < 1e-12
    jd = j.copy(); jd[...,0,1]=0; jd[...,1,0]=0
    _, ii = forward(jd, p[:3])
    assert np.max(np.abs(ii[2]-(ii[0]+ii[1])/2)) < 1e-10
    assert np.max(np.abs(u[2]-(u[0]+u[1])/np.sqrt(2))) < 1e-10
    assert np.max(np.abs(forward(j*np.exp(1j*.7),p)[1]-intensity)) < 1e-10
    impossible = np.zeros((3,2,2))
    impossible[2,0,0] = 1
    assert exact_zero_conflicts(impossible, p[:3]) == 1
    ys, xs = roi_slices(n)
    constructed_target = intensity[:3, ys, xs]
    projected = project_intensity(u[:3], constructed_target, 41)
    assert np.max(np.abs(projected-u[:3])) < 1e-10
    task = make_task(0, 'demo', n, .30, 20261122, 'constructed')
    _, known_intensity = forward(task['known_j'], task['polarization'])
    assert np.max(np.abs(known_intensity[:,ys,xs]-task['target_intensity'])) < 1e-4
    for z in (np.zeros((2,2), complex), np.eye(2, dtype=complex),
              np.array([[1,1j],[1j,1]],complex),
              np.array([[1,1e-12j],[1e-12j,1]],complex)):
        jj, _, v = takagi_project(z)
        assert np.max(np.abs(jj-jj.T)) < 1e-9
        assert np.max(np.abs(jj.conj().T@jj-np.eye(2))) < 1e-9
    for _ in range(100):
        z = rng.normal(size=(2,2))+1j*rng.normal(size=(2,2))
        z = (z+z.T)/2
        jj, singular, v = takagi_project(z)
        assert np.max(np.abs(v.conj().T@v-np.eye(2))) < 1e-9
        assert np.max(np.abs(v@np.diag(singular)@v.T-z)) < 1e-9
        assert np.max(np.abs(jj.conj().T@jj-np.eye(2))) < 1e-9
        candidate = random_jones(1, int(rng.integers(1, 1e9)), np.complex128)[0,0]
        assert np.linalg.norm(z-jj) <= np.linalg.norm(z-candidate)+1e-9
    out = ROOT/'reports'; out.mkdir(exist_ok=True)
    roundtrip = out/'validation_roundtrip.npz'
    np.savez_compressed(roundtrip, jones=j, polarization=p, intensity=intensity)
    with np.load(roundtrip, allow_pickle=False) as loaded:
        assert np.max(np.abs(forward(loaded['jones'], loaded['polarization'])[1]-loaded['intensity'])) < 1e-10
    roundtrip.unlink()
    (out/'validation.json').write_text(json.dumps(dict(status='passed', n=n,
        random_takagi_cases=100, max_power_error=float(np.max(np.abs(intensity.sum((1,2))-n*n))),
        python=platform.python_version(), numpy=np.__version__), indent=2), encoding='utf-8')
    print('physics validation passed')


def audit(path):
    rows = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
    checked = 0; failed = []
    for row in rows:
        if 'path' not in row: continue
        file = ROOT/row['path']
        if hashlib.sha256(file.read_bytes()).hexdigest() != row['sha256']:
            failed.append((row['task_id'], 'sha256')); continue
        with np.load(file, allow_pickle=False) as z:
            _, i = forward(z['jones'], z['polarization'])
            delta = float(np.max(np.abs(i-z['reconstruction_full'])))
            if delta > 1e-3: failed.append((row['task_id'], 'reconstruction %.3g' % delta))
            checked += 1
    print(json.dumps(dict(rows=len(rows), checked=checked, failed=failed), indent=2))
    if failed: raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('validate')
    gen = sub.add_parser('generate')
    gen.add_argument('--name', required=True)
    gen.add_argument('--split', default='dev', choices=['demo','dev','train','val','test','challenge'])
    gen.add_argument('--tasks', type=int, required=True)
    gen.add_argument('--seeds', type=int, default=4)
    gen.add_argument('--n', type=int)
    gen.add_argument('--beta', type=float)
    gen.add_argument('--constructed-every', type=int, default=5)
    gen.add_argument('--walltime-hours', type=float, default=8)
    gen.add_argument('--accepted-target', type=int)
    aud = sub.add_parser('audit'); aud.add_argument('manifest', type=Path)
    args = parser.parse_args()
    if args.command == 'validate': validate()
    elif args.command == 'generate': run(args)
    else: audit(args.manifest)


if __name__ == '__main__': main()
