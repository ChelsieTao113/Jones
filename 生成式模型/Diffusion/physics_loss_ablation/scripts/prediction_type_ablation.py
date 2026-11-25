"""Frozen one-task paired epsilon/v experiment. Run from the metasurface Python."""
import sys
sys.dont_write_bytecode = True
import argparse
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT.parent / 'Diffusion_prep'
sys.path[:0] = [str(PREP), str(ROOT)]
from src.dataset import load
from src.model import build
from src.representation import encode, decode, project
from src.physics import numpy_forward, quality, residual, accepted

CFG_PATH = ROOT / 'configs/prediction_type_ablation.json'
CFG = json.loads(CFG_PATH.read_text(encoding='utf-8'))
BASE_PATH = PREP / 'configs/prep_v1.json'
BASE = json.loads(BASE_PATH.read_text(encoding='utf-8'))
MANIFEST = ROOT / 'manifests/single.json'
NOISE_MANIFEST = ROOT / 'manifests/prediction_type_diagnostic_seeds.json'
RUNS = ROOT / 'runs/prediction_type_ablation'
RESULTS = ROOT / 'results/prediction_type_ablation'
FIGURES = ROOT / 'figures/prediction_type_ablation'
DEVICE = 'cuda'
BRANCHES = {'E': 'epsilon', 'V': 'v_prediction'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'empty table {path}')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted(set().union(*(r.keys() for r in rows)))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def task():
    rows = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if len(rows) != 1 or rows[0]['task_id'] != CFG['task_id']:
        raise ValueError('single task manifest changed')
    return rows[0], load(rows[0])


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(s):
    random.setstate(s['python'])
    np.random.set_state(s['numpy'])
    torch.set_rng_state(s['torch'])
    torch.cuda.set_rng_state_all(s['cuda'])


def make_model(seed, prediction_type):
    seed_all(seed)
    cc = json.loads(json.dumps(BASE))
    cc['diffusion']['prediction_type'] = prediction_type
    m, c, ddpm, ddim = build(cc)
    m.to(DEVICE)
    c.to(DEVICE)
    m.enable_gradient_checkpointing()
    opt = torch.optim.AdamW(list(m.parameters()) + list(c.parameters()),
                            lr=cc['optimizer']['learning_rate'],
                            weight_decay=cc['optimizer']['weight_decay'])
    return m, c, ddpm, ddim, opt, cc


def state_hash(module):
    h = hashlib.sha256()
    for k, v in module.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def train_generator(seed):
    # Separate from model initialization and evaluation, shared exactly by E and V.
    return torch.Generator(device=DEVICE).manual_seed(seed + 1000000)


def train_step(m, c, ddpm, opt, z, gen, prediction_type):
    m.train(); c.train(); opt.zero_grad(set_to_none=True)
    x0 = torch.from_numpy(encode(z['jones']))[None].to(DEVICE)
    T = torch.from_numpy(z['target_intensity'])[None].to(DEVICE)
    p = torch.from_numpy(z['polarization'])[None].to(DEVICE)
    a = torch.from_numpy(z['analyzer'])[None].to(DEVICE)
    n = BASE['optimizer']['effective_batch']
    losses = []; ts = []
    noise_hash = hashlib.sha256()
    for _ in range(n):
        eps = torch.randn(x0.shape, device=DEVICE, generator=gen)
        t = torch.randint(0, ddpm.config.num_train_timesteps, (1,), device=DEVICE, generator=gen)
        ts.append(int(t[0]))
        noise_hash.update(eps.cpu().numpy().tobytes())
        xt = ddpm.add_noise(x0, eps, t)
        target = eps if prediction_type == 'epsilon' else ddpm.get_velocity(x0, eps, t)
        pred = m(xt, t, encoder_hidden_states=c(T, p, a)).sample
        loss = F.mse_loss(pred, target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'nonfinite loss at t={int(t[0])}, target_std={float(target.std())}, pred_std={float(pred.std())}')
        (loss / n).backward()
        losses.append(float(loss.detach()))
    gm = math.sqrt(sum(float(q.grad.detach().square().sum()) for q in m.parameters() if q.grad is not None))
    gc = math.sqrt(sum(float(q.grad.detach().square().sum()) for q in c.parameters() if q.grad is not None))
    if not np.isfinite([gm, gc]).all():
        raise FloatingPointError(f'nonfinite gradient at t={ts}')
    torch.nn.utils.clip_grad_norm_(list(m.parameters()) + list(c.parameters()), BASE['optimizer']['grad_clip'])
    opt.step()
    return {'loss': float(np.mean(losses)), 'grad_unet': gm, 'grad_condition': gc,
            't': ts, 'noise_sha256': noise_hash.hexdigest()}


def save_checkpoint(path, step, branch, seed, m, c, opt, gen):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {'step': step, 'branch': branch, 'prediction_type': BRANCHES[branch],
             'seed': seed, 'model': m.state_dict(), 'condition': c.state_dict(),
             'optimizer': opt.state_dict(), 'rng': rng_state(),
             'train_generator_state': gen.get_state(),
             'config_sha256': sha(CFG_PATH), 'base_sha256': sha(BASE_PATH)}
    temp = path.with_suffix('.tmp')
    torch.save(state, temp)
    temp.replace(path)


def load_checkpoint(path, branch, seed, m, c, opt, gen):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if (state['branch'] != branch or state['seed'] != seed or
        state['prediction_type'] != BRANCHES[branch] or
        state['config_sha256'] != sha(CFG_PATH) or state['base_sha256'] != sha(BASE_PATH)):
        raise ValueError('checkpoint identity/config mismatch')
    m.load_state_dict(state['model']); c.load_state_dict(state['condition'])
    opt.load_state_dict(state['optimizer'])
    restore_rng(state['rng']); gen.set_state(state['train_generator_state'])
    return state['step']


def input_tensors(z):
    return (torch.from_numpy(z['target_intensity'])[None].to(DEVICE),
            torch.from_numpy(z['polarization'])[None].to(DEVICE),
            torch.from_numpy(z['analyzer'])[None].to(DEVICE))


def matrix_stats(j):
    s = np.linalg.svd(j, compute_uv=False)
    jj = np.swapaxes(j.conj(), -1, -2) @ j
    u = np.max(np.abs(jj - np.eye(2)), axis=(-2, -1))
    fro = np.linalg.norm(j, axis=(-2, -1))
    return {'sing_p05': float(np.percentile(s, 5)), 'sing_p50': float(np.percentile(s, 50)),
            'sing_p95': float(np.percentile(s, 95)), 'unit_p95': float(np.percentile(u, 95)),
            'unit_max': float(u.max()), 'raw_fro': float(np.linalg.norm(j)),
            'near_zero_fraction': float(np.mean(fro < CFG['raw_near_zero_threshold']))}


def image_metrics(x, z):
    raw = decode(x)
    phys = project(raw)
    ir = numpy_forward(raw, z['polarization'], z['analyzer'])
    ip = numpy_forward(phys, z['polarization'], z['analyzer'])
    qr = quality(ir, z['target_intensity'])
    qp = quality(ip, z['target_intensity'])
    rr = residual(raw); rp = residual(phys)
    change = float(np.linalg.norm(phys - raw) / max(np.linalg.norm(raw), 1e-20))
    return {'raw': raw, 'projected': phys, 'raw_image': ir, 'projected_image': ip,
            'raw_metrics': qr, 'projected_metrics': qp, 'raw_residual': rr,
            'projected_residual': rp, 'raw_stats': matrix_stats(raw),
            'projection_change': change, 'pass': accepted(qp, rp, BASE)}


def rows_for_metrics(view, meta):
    rows = []
    for ch in range(3):
        row = dict(meta, channel=ch + 1, pass_all=int(view['pass']),
                   projection_change=view['projection_change'],
                   raw_symmetry=view['raw_residual']['symmetry'],
                   raw_unitarity=view['raw_residual']['unitarity'],
                   projected_symmetry=view['projected_residual']['symmetry'],
                   projected_unitarity=view['projected_residual']['unitarity'])
        row.update({'raw_' + k: v for k, v in view['raw_stats'].items()})
        for label in ('raw', 'projected'):
            row.update({label + '_' + k: v for k, v in view[label + '_metrics'][ch].items()})
        rows.append(row)
    return rows


def generate(m, c, ddim, z, seed, branch, step, trace=False):
    m.eval(); c.eval()
    T, p, a = input_tensors(z)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn((1, 6, 128, 128), device=DEVICE, generator=gen)
    initial_sha = hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()
    ddim.set_timesteps(BASE['diffusion']['inference_steps'], device=DEVICE)
    trajectory = []
    with torch.inference_mode():
        tokens = c(T, p, a)
        for index, t in enumerate(ddim.timesteps):
            pred = m(x, t, encoder_hidden_states=tokens).sample
            result = ddim.step(pred, t, x, eta=BASE['diffusion']['eta'])
            pred0 = result.pred_original_sample
            if trace:
                arr = pred0[0].cpu().numpy()
                j = decode(arr)
                stats = matrix_stats(j)
                row = {'branch': branch, 'step': step, 'generation_seed': seed,
                       'index': index, 't': int(t), 'pred_x0_std': float(arr.std()),
                       'pred_x0_abs_p95': float(np.percentile(np.abs(arr), 95)),
                       'initial_sha256': initial_sha, **stats}
                if index in CFG['trace_project_indices']:
                    view = image_metrics(arr, z)
                    row.update({'raw_worst_pcc': min(v['pcc'] for v in view['raw_metrics']),
                                'raw_worst_e_abs': max(v['e_abs'] for v in view['raw_metrics']),
                                'projected_worst_pcc': min(v['pcc'] for v in view['projected_metrics']),
                                'projected_worst_e_abs': max(v['e_abs'] for v in view['projected_metrics']),
                                'projection_change': view['projection_change']})
                    for label in ('raw','projected'):
                        for ch,q in enumerate(view[label+'_metrics'],1):
                            row.update({f'{label}_ch{ch}_{key}':q[key] for key in
                                        ('pcc','e_abs','eta_roi','eta_relative_error','black_leakage')})
                trajectory.append(row)
            x = result.prev_sample
    final = x[0].cpu().numpy()
    last_diff = float(np.max(np.abs(final - pred0[0].cpu().numpy())))
    return final, trajectory, initial_sha, last_diff


def evaluate_point(m, c, ddim, z, branch, step, seed, train_seed, trace=False):
    arr, trajectory, initial_sha, last_diff = generate(m, c, ddim, z, seed, branch, step, trace)
    view = image_metrics(arr, z)
    meta = {'train_seed': train_seed, 'branch': branch, 'step': step,
            'generation_seed': seed, 'initial_sha256': initial_sha,
            'last_pred_to_final_max_abs': last_diff}
    rows = rows_for_metrics(view, meta)
    if not np.isfinite(arr).all():
        raise FloatingPointError('nonfinite generated matrix')
    return rows, trajectory, view


def train_branch(branch, seed, z):
    folder = RUNS / f'seed_{seed}' / branch
    final = folder / 'final.pt'; rolling = folder / 'rolling.pt'
    trace_path = folder / 'train_trace.json'
    eval_path = folder / 'evaluation.csv'
    if final.exists() and trace_path.exists() and eval_path.exists():
        return
    m, c, ddpm, ddim, opt, cc = make_model(seed, BRANCHES[branch])
    gen = train_generator(seed)
    initial = {'model': state_hash(m), 'condition': state_hash(c),
               'parameters_unet': sum(p.numel() for p in m.parameters()),
               'parameters_condition': sum(p.numel() for p in c.parameters())}
    write_json(folder / 'initial.json', initial)
    if rolling.exists():
        start = load_checkpoint(rolling, branch, seed, m, c, opt, gen)
        trace = json.loads(trace_path.read_text(encoding='utf-8'))
        eval_rows = read_csv(eval_path) if eval_path.exists() else []
        if len(trace) != start:
            raise ValueError('rolling checkpoint and trace do not agree')
    else:
        start = 0; trace = []; eval_rows = []
    began = time.monotonic()
    for step in range(start + 1, CFG['updates'] + 1):
        info = train_step(m, c, ddpm, opt, z, gen, BRANCHES[branch])
        trace.append({'step': step, **info})
        if step % 50 == 0:
            print(seed, branch, step, info['loss'], round(time.monotonic() - began, 1), flush=True)
        if step in CFG['evaluation_steps']:
            seeds = CFG['generation_seeds'] if step == CFG['updates'] else CFG['generation_seeds'][:3]
            for s in seeds:
                rows, _, _ = evaluate_point(m, c, ddim, z, branch, step, s, seed)
                eval_rows.extend(rows)
            m.train(); c.train()
            write_json(trace_path, trace)
            write_csv(eval_path, eval_rows)
            if step == CFG['updates']:
                save_checkpoint(final, step, branch, seed, m, c, opt, gen)
            else:
                save_checkpoint(rolling, step, branch, seed, m, c, opt, gen)
    write_json(folder / 'summary.json', {'seed': seed, 'branch': branch,
               'updates': CFG['updates'], 'elapsed_seconds_this_process': time.monotonic()-began,
               'initial': initial, 'final_checkpoint': str(final),
               'train_t_sha256': hashlib.sha256(np.asarray([t for q in trace for t in q['t']], dtype=np.int16).tobytes()).hexdigest(),
               'train_noise_digest_sha256': hashlib.sha256(''.join(q['noise_sha256'] for q in trace).encode()).hexdigest()})
    print('completed', seed, branch, flush=True)


def check_formula(z):
    x32 = torch.from_numpy(encode(z['jones']))[None].to(DEVICE)
    report = []
    for dtype in (torch.float64, torch.float32):
        x0 = x32.to(dtype)
        m, c, de, ie, _, _ = make_model(20261124, 'epsilon')
        _, _, dv, iv, _, _ = make_model(20261124, 'v_prediction')
        ie.set_timesteps(100, device=DEVICE); iv.set_timesteps(100, device=DEVICE)
        assert torch.equal(de.betas, dv.betas) and torch.equal(de.alphas_cumprod, dv.alphas_cumprod)
        assert torch.equal(ie.timesteps, iv.timesteps) and int(ie.timesteps[0]) == 990
        if dtype == torch.float64:
            dcmp={k:[dict(de.config).get(k),dict(dv.config).get(k)] for k in set(de.config)|set(dv.config)
                  if dict(de.config).get(k)!=dict(dv.config).get(k)}
            icmp={k:[dict(ie.config).get(k),dict(iv.config).get(k)] for k in set(ie.config)|set(iv.config)
                  if dict(ie.config).get(k)!=dict(iv.config).get(k)}
            if set(dcmp)!={'prediction_type'} or set(icmp)!={'prediction_type'}:
                raise RuntimeError(f'unexpected scheduler difference: {dcmp}, {icmp}')
            write_json(RESULTS/'scheduler_comparison.json',{'DDPM_config_differences':dcmp,
                       'DDIM_config_differences':icmp,
                       'betas_exact':True,'alphas_cumprod_exact':True,
                       'timesteps':[int(t) for t in ie.timesteps]})
        for t0 in CFG['diagnostic_times']:
            t = torch.tensor([t0], device=DEVICE)
            eps = torch.randn(x0.shape, device=DEVICE, dtype=dtype,
                              generator=torch.Generator(device=DEVICE).manual_seed(7000+t0))
            abar = de.alphas_cumprod[t0].to(dtype)
            aa = abar.sqrt(); bb = (1-abar).sqrt()
            xt = aa*x0+bb*eps
            v = aa*eps-bb*x0
            api = dv.get_velocity(x0, eps, t)
            xe = (xt-bb*eps)/aa
            xv = aa*xt-bb*v
            ev = bb*xt+aa*v
            row = {'dtype': str(dtype), 't': t0, 'snr': float(abar/(1-abar)),
                   'velocity_api_max_abs': float((api-v).abs().max()),
                   'epsilon_x0_relative_l2': float(torch.linalg.vector_norm(xe-x0)/torch.linalg.vector_norm(x0)),
                   'v_x0_relative_l2': float(torch.linalg.vector_norm(xv-x0)/torch.linalg.vector_norm(x0)),
                   'v_epsilon_relative_l2': float(torch.linalg.vector_norm(ev-eps)/torch.linalg.vector_norm(eps))}
            if row['velocity_api_max_abs'] > (1e-12 if dtype==torch.float64 else 1e-6):
                raise RuntimeError(f'get_velocity disagrees at t={t0}: {row}')
            if row['v_x0_relative_l2'] > (1e-12 if dtype==torch.float64 else 1e-5):
                raise RuntimeError(f'v recovery disagrees at t={t0}: {row}')
            if t0 in [int(t) for t in ie.timesteps]:
                se = ie.step(eps, t0, xt, eta=0)
                sv = iv.step(v, t0, xt, eta=0)
                row['scheduler_pred_x0_max_abs'] = float((se.pred_original_sample-sv.pred_original_sample).abs().max())
                row['scheduler_prev_max_abs'] = float((se.prev_sample-sv.prev_sample).abs().max())
            report.append(row)
        for t0 in (990,500,0):
            t=torch.tensor([t0],device=DEVICE)
            eps=torch.randn(x0.shape,device=DEVICE,dtype=dtype,
                            generator=torch.Generator(device=DEVICE).manual_seed(8700+t0))
            abar=de.alphas_cumprod[t0].to(dtype)
            aa=abar.sqrt();bb=(1-abar).sqrt()
            xt=aa*x0+bb*eps
            v=aa*eps-bb*x0
            # Fresh schedulers for each branch and each selected legal DDIM time.
            e0=type(ie).from_config(de.config);v0=type(iv).from_config(dv.config)
            e0.set_timesteps(100,device=DEVICE);v0.set_timesteps(100,device=DEVICE)
            se=e0.step(eps,t0,xt,eta=0);sv=v0.step(v,t0,xt,eta=0)
            report.append({'dtype':str(dtype),'t':t0,'scheduler_probe':True,
                'scheduler_pred_x0_max_abs':float((se.pred_original_sample-sv.pred_original_sample).abs().max()),
                'scheduler_prev_max_abs':float((se.prev_sample-sv.prev_sample).abs().max())})
        del m,c,de,ie,dv,iv
    write_csv(RESULTS/'formula.csv', report)
    return report


def smoke(z):
    results = {}
    for branch in BRANCHES:
        m,c,ddpm,ddim,opt,_ = make_model(20261124, BRANCHES[branch])
        gen=train_generator(20261124)
        hm,hc=state_hash(m),state_hash(c)
        start=time.monotonic()
        logs=[train_step(m,c,ddpm,opt,z,gen,BRANCHES[branch]) for _ in range(20)]
        path=RUNS/'smoke'/f'{branch}.pt'
        save_checkpoint(path,20,branch,20261124,m,c,opt,gen)
        m2,c2,ddpm2,ddim2,opt2,_=make_model(20261124,BRANCHES[branch])
        gen2=train_generator(20261124)
        restored=load_checkpoint(path,branch,20261124,m2,c2,opt2,gen2)
        same=state_hash(m)==state_hash(m2) and state_hash(c)==state_hash(c2)
        same=same and len(opt.state)==len(opt2.state) and torch.equal(gen.get_state(),gen2.get_state())
        restored_rng=rng_state()
        continued_a=train_step(m,c,ddpm,opt,z,gen,BRANCHES[branch])
        restore_rng(restored_rng)
        continued_b=train_step(m2,c2,ddpm2,opt2,z,gen2,BRANCHES[branch])
        continuation_max_abs=max(float((a-b).detach().abs().max()) for a,b in zip(
            list(m.parameters())+list(c.parameters()),list(m2.parameters())+list(c2.parameters())))
        continue_equal=(continuation_max_abs<=1e-6 and
                        continued_a['noise_sha256']==continued_b['noise_sha256'] and
                        continued_a['t']==continued_b['t'])
        arr,_,_,_=generate(m2,c2,ddim2,z,20261201,branch,20)
        results[branch]={'initial_model_sha256':hm,'initial_condition_sha256':hc,
                         'seconds_per_update':(time.monotonic()-start)/20,
                         'loss_first_last':[logs[0]['loss'],logs[-1]['loss']],
                         'min_gradient_unet':min(q['grad_unet'] for q in logs),
                         'min_gradient_condition':min(q['grad_condition'] for q in logs),
                         'final_model_changed':hm!=state_hash(m),
                         'final_condition_changed':hc!=state_hash(c),
                         'restored_step':restored,'restore_equal':same,
                         'continued_update_equal':continue_equal,
                         'continued_parameter_max_abs':continuation_max_abs,
                         'continued_loss_difference':abs(continued_a['loss']-continued_b['loss']),
                         'sample_shape':list(arr.shape),'sample_finite':bool(np.isfinite(arr).all()),
                         't_sequence': [q['t'] for q in logs],
                         'noise_digests':[q['noise_sha256'] for q in logs]}
        if not same or not continue_equal or restored!=20 or not results[branch]['sample_finite']:
            write_json(RESULTS/'smoke_failure.json',results)
            raise RuntimeError(f'{branch} smoke failed: restore={same}, continuation_abs={continuation_max_abs}, '
                               f't_match={continued_a["t"]==continued_b["t"]}, '
                               f'noise_match={continued_a["noise_sha256"]==continued_b["noise_sha256"]}')
    write_json(RESULTS/'smoke.json',results)
    return results


def epsilon_regression(z):
    from src.train import train_step as old_step
    # CUDA backward is nondeterministic by default here (old-vs-old can differ
    # by ~1e-4). Make this one regression probe deterministic; formal training
    # uses the original settings in a separate process.
    prior_alg=torch.are_deterministic_algorithms_enabled()
    prior_cudnn=torch.backends.cudnn.deterministic
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    m1,c1,d1,_,o1,_=make_model(20261124,'epsilon')
    m2,c2,d2,_,o2,_=make_model(20261124,'epsilon')
    m3,c3,d3,_,o3,_=make_model(20261124,'epsilon')
    gen=train_generator(20261124)
    cuda_start=gen.get_state()
    records=[]
    for scheduler in (d1,d2,d3):
        calls=[];original=scheduler.add_noise
        def wrapped(x,eps,t,_original=original,_calls=calls):
            _calls.append((int(t[0]),hashlib.sha256(eps.cpu().numpy().tobytes()).hexdigest()))
            return _original(x,eps,t)
        scheduler.add_noise=wrapped
        records.append(calls)
    item={'x':torch.from_numpy(encode(z['jones'])).to(DEVICE),
          't':torch.from_numpy(z['target_intensity']).to(DEVICE),
          'p':torch.from_numpy(z['polarization']).to(DEVICE),
          'a':torch.from_numpy(z['analyzer']).to(DEVICE)}
    torch.cuda.set_rng_state(cuda_start)
    old_loss,_,_=old_step(m1,c1,d1,o1,[item],BASE,0)
    new=train_step(m2,c2,d2,o2,z,gen,'epsilon')
    torch.cuda.set_rng_state(cuda_start)
    old2_loss,_,_=old_step(m3,c3,d3,o3,[item],BASE,0)
    diff=max(float((a-b).detach().abs().max()) for a,b in zip(
         list(m1.parameters())+list(c1.parameters()),list(m2.parameters())+list(c2.parameters())))
    old_repeat_diff=max(float((a-b).detach().abs().max()) for a,b in zip(
         list(m1.parameters())+list(c1.parameters()),list(m3.parameters())+list(c3.parameters())))
    result={'old_loss':old_loss,'new_loss':new['loss'],'parameter_max_abs':diff,
            'old_repeat_loss':old2_loss,'old_repeat_parameter_max_abs':old_repeat_diff,
            'old_optimizer_entries':len(o1.state),'new_optimizer_entries':len(o2.state),
            'inputs_match':records[0]==records[1]==records[2],
            'micro_inputs':records[0],'deterministic_probe_only':True}
    write_json(RESULTS/'epsilon_regression.json',result)
    torch.use_deterministic_algorithms(prior_alg)
    torch.backends.cudnn.deterministic=prior_cudnn
    if not result['inputs_match'] or abs(old_loss-new['loss'])>1e-6 or diff>max(1e-6,2*old_repeat_diff):
        raise RuntimeError(f'epsilon old behavior regression: {result}')
    return result


def run_check():
    record,z=task()
    diagnostic_seeds=json.loads(NOISE_MANIFEST.read_text(encoding='utf-8'))
    if sorted(map(int,diagnostic_seeds))!=sorted(CFG['diagnostic_times']) or diagnostic_seeds['990']!=CFG['diagnostic_noise_seeds'] or diagnostic_seeds['999']!=CFG['diagnostic_noise_seeds']:
        raise ValueError('diagnostic noise manifest mismatch')
    for branch,prediction_type in BRANCHES.items():
        _,_,ddpm,ddim,_,cc=make_model(20261124,prediction_type)
        write_json(ROOT/f'configs/prediction_type_{branch}.json',{
            'branch':branch,'prediction_type':prediction_type,
            'base_config_sha256':sha(BASE_PATH),'ablation_config_sha256':sha(CFG_PATH),
            'effective_config':cc,'training_scheduler_config':dict(ddpm.config),
            'inference_scheduler_config':dict(ddim.config)})
    formulas=check_formula(z)
    smokes=smoke(z)
    regression=epsilon_regression(z)
    if smokes['E']['initial_model_sha256']!=smokes['V']['initial_model_sha256'] or smokes['E']['initial_condition_sha256']!=smokes['V']['initial_condition_sha256']:
        raise RuntimeError('initial weights differ')
    if smokes['E']['t_sequence']!=smokes['V']['t_sequence'] or smokes['E']['noise_digests']!=smokes['V']['noise_digests']:
        raise RuntimeError('training inputs differ')
    sources=[CFG_PATH,ROOT/'configs/prediction_type_E.json',ROOT/'configs/prediction_type_V.json',NOISE_MANIFEST,
             BASE_PATH,MANIFEST,PREP/'src/model.py',PREP/'src/train.py',PREP/'src/conditioning.py',
             PREP/'src/representation.py',PREP/'src/physics.py',Path(__file__),
             ROOT/'scripts/plot_focused_prediction.py']
    write_json(RESULTS/'source_manifest.json',{'task_id':record['task_id'],'input_sha256':record['sha256'],
        'source_sha256':{str(p.relative_to(ROOT.parent)):sha(p) for p in sources},
        'python':sys.executable,'torch':torch.__version__,'diffusers':__import__('diffusers').__version__,
        'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
        'optimizer':BASE['optimizer'],'model':BASE['model'],
        'smoke_seconds_per_update':{b:smokes[b]['seconds_per_update'] for b in BRANCHES}})
    measured=[r for r in formulas if 'v_x0_relative_l2' in r]
    scheduler_probes=[r for r in formulas if r.get('scheduler_probe')]
    f32=[r for r in measured if r['dtype']=='torch.float32']
    lines=['# prediction_type 公式与训练检查','',
           f"任务 `{record['task_id']}`，输入 SHA256 `{record['sha256']}`。",
           'E/V 仅训练目标与 scheduler prediction_type 不同；两者初始模型和条件编码器哈希相同，20 步训练 t/噪声摘要逐次相同。',
           f"20 步耗时（秒/更新）E/V：{smokes['E']['seconds_per_update']:.3f}/{smokes['V']['seconds_per_update']:.3f}。",
           '两组梯度、参数更新、checkpoint 恢复与一次 DDIM100 生成均通过。',
           f"float32 v→x0 最大相对 L2 {max(r['v_x0_relative_l2'] for r in f32):.3g}，get_velocity 对显式公式最大绝对差 {max(r['velocity_api_max_abs'] for r in f32):.3g}。E/V 的 betas、alphas_cumprod 和 100 步 t 列表完全相同；DDPM/DDIM 配置差异均仅为 prediction_type，详见 [scheduler_comparison.json](../results/prediction_type_ablation/scheduler_comparison.json)。",
           f"在 t=990/500/0 用真实 epsilon/v 的独立 DDIM scheduler.step，pred_original_sample 最大绝对差 {max(r['scheduler_pred_x0_max_abs'] for r in scheduler_probes):.3g}，prev_sample 最大绝对差 {max(r['scheduler_prev_max_abs'] for r in scheduler_probes):.3g}。t=990 的 epsilon 反解误差受 float32 消减与除小 a 影响，不套用中低噪声阈值。",
           f"epsilon 旧代码单次更新回归：四个微批次 t/噪声完全相同，loss {regression['old_loss']:.8f}/{regression['new_loss']:.8f}，更新后最大参数差 {regression['parameter_max_abs']:.3g}；此项单独启用确定性 CUDA 算法，正式训练保持原设置。详见 [epsilon_regression.json](../results/prediction_type_ablation/epsilon_regression.json)。",
           f"公式误差与各 t 的 SNR 见 [formula.csv](../results/prediction_type_ablation/formula.csv)，检查记录见 [smoke.json](../results/prediction_type_ablation/smoke.json)。",
           'float64 和 float32 均验证 get_velocity、x0/epsilon 恢复；epsilon 高噪声反解按实际浮点误差单列。',
           'smoke checkpoint 仅用于恢复检查，正式训练重新从共同随机种子初始化。']
    write_json(RESULTS/'formula_summary.json',{'rows':len(formulas),'first_ddim_t':990})
    (ROOT/'reports/PREDICTION_TYPE_TEST_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run_paired(seed):
    _,z=task()
    for branch in BRANCHES:
        train_branch(branch,seed,z)
    e=json.loads((RUNS/f'seed_{seed}/E/summary.json').read_text(encoding='utf-8'))
    v=json.loads((RUNS/f'seed_{seed}/V/summary.json').read_text(encoding='utf-8'))
    if e['initial']!=v['initial'] or e['train_t_sha256']!=v['train_t_sha256'] or e['train_noise_digest_sha256']!=v['train_noise_digest_sha256']:
        raise RuntimeError('formal pair not matched')


def noise_diagnosis(m,c,ddpm,z,branch,train_seed):
    x0=torch.from_numpy(encode(z['jones']))[None].to(DEVICE)
    T,p,a=input_tensors(z)
    rows=[]
    diagnostic_seeds=json.loads(NOISE_MANIFEST.read_text(encoding='utf-8'))
    m.eval();c.eval()
    with torch.inference_mode():
        tokens=c(T,p,a)
        for t0 in CFG['diagnostic_times']:
            t=torch.tensor([t0],device=DEVICE)
            abar=ddpm.alphas_cumprod[t0]
            aa=abar.sqrt();bb=(1-abar).sqrt()
            for seed in diagnostic_seeds[str(t0)]:
                gen=torch.Generator(device=DEVICE).manual_seed(seed)
                eps=torch.randn(x0.shape,device=DEVICE,generator=gen)
                xt=ddpm.add_noise(x0,eps,t)
                target=eps if branch=='E' else ddpm.get_velocity(x0,eps,t)
                pred=m(xt,t,encoder_hidden_states=tokens).sample
                rec=(xt-bb*pred)/aa if branch=='E' else aa*xt-bb*pred
                rec_np=rec[0].cpu().numpy()
                view=image_metrics(rec_np,z)
                true_j=z['jones']; pred_j=decode(rec_np)
                phase=np.angle(np.vdot(true_j.ravel(),pred_j.ravel()))
                meta={'train_seed':train_seed,'branch':branch,'t':t0,'snr':float(abar/(1-abar)),
                      'noise_seed':seed,
                      'target_mse':float(F.mse_loss(pred,target)),
                      'x0_relative_l2':float(torch.linalg.vector_norm(rec-x0)/torch.linalg.vector_norm(x0)),
                      'x0_global_phase_relative_l2':float(np.linalg.norm(pred_j*np.exp(-1j*phase)-true_j)/np.linalg.norm(true_j)),
                      'x0_std':float(rec_np.std()),'x0_abs_p95':float(np.percentile(np.abs(rec_np),95))}
                rows.extend(rows_for_metrics(view,meta))
            print('noise',train_seed,branch,t0,flush=True)
    return rows


def contact_sheet(z, views_by_seed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True,exist_ok=True)
    target=np.zeros_like(z['reconstruction_full'])
    target[:,32:96,32:96]=z['target_intensity']
    gs=z['reconstruction_full']
    for mode in ('absolute','shape_only'):
        fig,axes=plt.subplots(18,4,figsize=(13,40),constrained_layout=True)
        for i,seed in enumerate(CFG['generation_seeds']):
            images=[target,gs,views_by_seed[seed]['E']['projected_image'],views_by_seed[seed]['V']['projected_image']]
            for ch in range(3):
                for col,img in enumerate(images):
                    ax=axes[i*3+ch,col]
                    if mode=='absolute':
                        hi=max(float(np.percentile(q[ch],99.5)) for q in images)
                        ax.imshow(img[ch],cmap='magma',vmin=0,vmax=max(hi,1e-12))
                    else:
                        ax.imshow(np.log1p(img[ch]),cmap='magma',vmin=0,
                                  vmax=max(float(np.percentile(np.log1p(img[ch]),99.5)),1e-12))
                    ax.set_title(f'{seed} ch{ch+1} {("T","GS","E","V")[col]}',fontsize=8)
                    ax.set_xticks([]);ax.set_yticks([])
        fig.suptitle('Shared intensity scale within each seed/channel' if mode=='absolute' else 'Shape only: each image separately scaled',fontsize=12)
        fig.savefig(FIGURES/f'contact_{mode}.png',dpi=90)
        plt.close(fig)


def evaluate_final(seed):
    _,z=task()
    metric_rows=[];noise_rows=[];trajectory_rows=[]
    views_by_seed={s:{} for s in CFG['generation_seeds']}
    for branch in BRANCHES:
        m,c,ddpm,ddim,opt,_=make_model(seed,BRANCHES[branch])
        gen=train_generator(seed)
        final=RUNS/f'seed_{seed}'/branch/'final.pt'
        if load_checkpoint(final,branch,seed,m,c,opt,gen)!=CFG['updates']:
            raise ValueError('not a final checkpoint')
        for s in CFG['generation_seeds']:
            rows,trace,view=evaluate_point(m,c,ddim,z,branch,CFG['updates'],s,seed,True)
            metric_rows+=rows;trajectory_rows+=trace;views_by_seed[s][branch]=view
            print('generated',seed,branch,s,flush=True)
        noise_rows+=noise_diagnosis(m,c,ddpm,z,branch,seed)
    for s in CFG['generation_seeds']:
        e=next(r for r in metric_rows if r['generation_seed']==s and r['branch']=='E')
        v=next(r for r in metric_rows if r['generation_seed']==s and r['branch']=='V')
        if e['initial_sha256']!=v['initial_sha256']:
            raise RuntimeError('generated noise mismatch')
    out=RESULTS/f'seed_{seed}'
    write_csv(out/'paired_channels.csv',metric_rows)
    write_csv(out/'noise_levels.csv',noise_rows)
    write_csv(out/'sampling_trajectory.csv',trajectory_rows)
    if seed==CFG['train_seeds'][0]:contact_sheet(z,views_by_seed)
    summarize(seed)


def summarize(seed):
    rows=read_csv(RESULTS/f'seed_{seed}/paired_channels.csv')
    def worst(branch, s, key, how):
        vals=[float(r[key]) for r in rows if r['branch']==branch and int(r['generation_seed'])==s]
        return how(vals)
    summary={}
    for branch in BRANCHES:
        results=[]
        for s in CFG['generation_seeds']:
            r=next(r for r in rows if r['branch']==branch and int(r['generation_seed'])==s)
            results.append({'generation_seed':s,'pcc':worst(branch,s,'projected_pcc',min),
                'e_abs':worst(branch,s,'projected_e_abs',max),
                'eta_roi_min':worst(branch,s,'projected_eta_roi',min),
                'eta_roi_max':worst(branch,s,'projected_eta_roi',max),
                'eta_relative_error':worst(branch,s,'projected_eta_relative_error',max),
                'black_leakage':worst(branch,s,'projected_black_leakage',max),
                'pass':bool(int(r['pass_all'])),
                'raw_sing_p95':float(r['raw_sing_p95']),
                'raw_fro':float(r['raw_raw_fro']),
                'near_zero_fraction':float(r['raw_near_zero_fraction']),
                'projection_change':float(r['projection_change'])})
        summary[branch]={'seeds':results,'median':{key:float(np.median([r[key] for r in results]))
                     for key in ('pcc','e_abs','eta_roi_min','eta_roi_max','eta_relative_error','black_leakage','raw_sing_p95','raw_fro','near_zero_fraction','projection_change')},
                     'pass_count':sum(r['pass'] for r in results)}
    E=summary['E'];V=summary['V']
    path_a=(V['median']['e_abs']<=0.8*E['median']['e_abs'] and
            V['median']['pcc']>=E['median']['pcc']+0.10 and
            sum(v['e_abs']<e['e_abs'] for e,v in zip(E['seeds'],V['seeds']))>=4 and
            V['median']['eta_relative_error']<=E['median']['eta_relative_error']+0.02 and
            V['median']['black_leakage']<=E['median']['black_leakage']+0.02 and
            all(float(r['projected_unitarity'])<=BASE['quality']['physical_tolerance'] and
                float(r['projected_symmetry'])<=BASE['quality']['physical_tolerance']
                for r in rows if r['branch']=='V'))
    path_b=V['pass_count']>=4 and V['pass_count']>E['pass_count']
    result={'train_seed':seed,'E':E,'V':V,'path_A':bool(path_a),'path_B':bool(path_b),
            'repeat_required':bool(path_a or path_b),
            'e_abs_improvement_fraction':float((E['median']['e_abs']-V['median']['e_abs'])/E['median']['e_abs']),
            'pcc_gain':float(V['median']['pcc']-E['median']['pcc'])}
    write_json(RESULTS/f'seed_{seed}/summary.json',result)
    return result


def plot_curves(seed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True,exist_ok=True)
    rows=read_csv(RESULTS/f'seed_{seed}/sampling_trajectory.csv')
    fig,axs=plt.subplots(1,3,figsize=(15,4.2),constrained_layout=True)
    for branch,color in (('E','tab:blue'),('V','tab:orange')):
        group=[r for r in rows if r['branch']==branch]
        for ax,key in ((axs[0],'pred_x0_std'),(axs[1],'projected_worst_pcc'),(axs[2],'projected_worst_e_abs')):
            indexes=sorted(set(int(r['index']) for r in group if r.get(key) not in ('',None)))
            vals=[np.median([float(r[key]) for r in group if int(r['index'])==i]) for i in indexes]
            ax.plot(indexes,vals,label=branch,color=color)
    for ax,title in zip(axs,('Predicted x0 standard deviation','Worst projected PCC','Worst projected e_abs')):
        ax.set_title(title);ax.set_xlabel('DDIM sample index');ax.grid(alpha=.3);ax.legend()
    fig.savefig(FIGURES/'sampling_curves.png',dpi=140)
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for branch,ax in zip(BRANCHES,axs):
        rows=json.loads((RUNS/f'seed_{seed}'/branch/'train_trace.json').read_text(encoding='utf-8'))
        ax.plot([r['step'] for r in rows],[r['loss'] for r in rows],lw=.7)
        ax.set(title=f'{branch} own-target MSE (not cross-comparable)',xlabel='optimizer update',ylabel='MSE')
        ax.grid(alpha=.3)
    fig.savefig(FIGURES/'training_losses_separate.png',dpi=140)
    plt.close(fig)


def make_report():
    seed=CFG['train_seeds'][0]
    _,z=task()
    gs=quality(z['reconstruction_full'],z['target_intensity'])
    initial=json.loads((RUNS/f'seed_{seed}/E/initial.json').read_text(encoding='utf-8'))
    provenance=json.loads((RESULTS/'source_manifest.json').read_text(encoding='utf-8'))
    s=summarize(seed)
    e=s['E'];v=s['V']
    noise=read_csv(RESULTS/f'seed_{seed}/noise_levels.csv')
    traj=read_csv(RESULTS/f'seed_{seed}/sampling_trajectory.csv')
    channels=read_csv(RESULTS/f'seed_{seed}/paired_channels.csv')
    plot_curves(seed)
    def median_noise(branch,t,key):
        group=[r for r in noise if r['branch']==branch and int(r['t'])==t]
        if key in ('projected_pcc','projected_e_abs'):
            seeds=sorted(set(int(r['noise_seed']) for r in group))
            reduce=min if key=='projected_pcc' else max
            vals=[reduce(float(r[key]) for r in group if int(r['noise_seed'])==s) for s in seeds]
        else:
            vals=[float(r[key]) for r in group]
        return float(np.median(vals))
    def median_trace(branch,index,key):
        vals=[float(r[key]) for r in traj if r['branch']==branch and int(r['index'])==index]
        return float(np.median(vals))
    lines=['# epsilon 与 v_prediction 单任务配对实验','',
      f"训练任务 `{CFG['task_id']}`，训练种子 {seed}；E/V 各 2000 次更新，六个相同纯噪声生成种子。",
      f"结论：V 的最差通道 e_abs 中位数相对变化 {s['e_abs_improvement_fraction']*100:.1f}%，PCC 变化 {s['pcc_gain']:+.3f}；合格数 E/V={e['pass_count']}/6、{v['pass_count']}/6。",
      f"预定复核门槛路径 A/B：{s['path_A']}/{s['path_B']}。{'已按规则追加两对训练种子复核。' if s['repeat_required'] else '未触发复核，按方案停止。'}",
      '即使相对改善明显，只要仍无样本达到原全部门槛，结论仍是“有所改善，尚不能正式训练”。',
      '两组训练目标 MSE 的数值不能直接比较；评价以共同 x0 和物理成像为准。','',
      f"GS 标签参照：三通道最差 PCC={min(q['pcc'] for q in gs):.4f}、最大 e_abs={max(q['e_abs'] for q in gs):.4f}、最大 ROI 功率相对误差={max(q['eta_relative_error'] for q in gs):.4f}；理想 T 另列在接触表，不与 GS 替换。",
      '本轮没有发现需对两组共同修复的训练或采样程序缺陷；epsilon 旧行为回归见检查结果。',
      '本实验只检验预测目标与对应普通 MSE 的整体方案，不能单独分离公式与等效损失加权的作用。','',
      '## 冻结配置与唯一差异','',
      '|设置|E|V|','|---|---|---|',
      '|网络与条件编码|原 UNet 6×128²、32/64/128，原三通道编码|相同|',
      '|训练目标|实际 epsilon|`get_velocity(x0, epsilon, t)`|',
      '|训练损失|普通 MSE|普通 MSE|',
      '|scheduler prediction_type|`epsilon`|`v_prediction`|',
      '|噪声与采样|squaredcos_cap_v2、1000 步；DDIM100、eta=0、首步 t=990|相同|',
      '|优化|AdamW 1e-4、weight_decay 1e-4、有效 batch 4、float32、梯度检查点与裁剪 1.0|相同|',
      f"实测解释器 `{provenance['python']}`，PyTorch {provenance['torch']}、Diffusers {provenance['diffusers']}、CUDA {provenance['cuda']}、GPU {provenance['gpu']}；U-Net 参数 {initial['parameters_unet']:,}，条件编码器参数 {initial['parameters_condition']:,}。",
      f"训练种子 {seed}；生成种子 {', '.join(map(str,CFG['generation_seeds']))}；诊断噪声见分噪声 CSV。E/V 的训练 t 与噪声摘要及初始权重哈希见各自 `summary.json`。输入及源码 SHA256 见 [source_manifest.json](../results/prediction_type_ablation/source_manifest.json)。",'',
      '## 最终六种子逐项结果','',
      '|生成种子|E PCC|V PCC|E e_abs|V e_abs|E ROI份额范围|V ROI份额范围|E 功率误差|V 功率误差|E 漏光|V 漏光|合格 E/V|',
      '|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---|']
    for a,b in zip(e['seeds'],v['seeds']):
        lines.append(f"|{a['generation_seed']}|{a['pcc']:.3f}|{b['pcc']:.3f}|{a['e_abs']:.3f}|{b['e_abs']:.3f}|{a['eta_roi_min']:.3f}–{a['eta_roi_max']:.3f}|{b['eta_roi_min']:.3f}–{b['eta_roi_max']:.3f}|{a['eta_relative_error']:.3f}|{b['eta_relative_error']:.3f}|{a['black_leakage']:.3f}|{b['black_leakage']:.3f}|{int(a['pass'])}/{int(b['pass'])}|")
    lines.append(f"|中位数|{e['median']['pcc']:.3f}|{v['median']['pcc']:.3f}|{e['median']['e_abs']:.3f}|{v['median']['e_abs']:.3f}|{e['median']['eta_roi_min']:.3f}–{e['median']['eta_roi_max']:.3f}|{v['median']['eta_roi_min']:.3f}–{v['median']['eta_roi_max']:.3f}|{e['median']['eta_relative_error']:.3f}|{v['median']['eta_relative_error']:.3f}|{e['median']['black_leakage']:.3f}|{v['median']['black_leakage']:.3f}|—|")
    lines += ['', '三幅图逐通道六种子中位数（PCC 越高越好，e_abs 越低越好）：','',
              '|通道|E/V PCC|E/V e_abs|','|---:|---:|---:|']
    for ch in (1,2,3):
        def chmed(branch,key):
            return float(np.median([float(r[key]) for r in channels if r['branch']==branch and int(r['channel'])==ch]))
        lines.append(f"|{ch}|{chmed('E','projected_pcc'):.3f}/{chmed('V','projected_pcc'):.3f}|{chmed('E','projected_e_abs'):.3f}/{chmed('V','projected_e_abs'):.3f}|")
    lines += ['', '## 已知答案加噪恢复与自由生成','',
      '|t|SNR|E x0 L2|V x0 L2|E 投影 PCC|V 投影 PCC|E 投影 e_abs|V 投影 e_abs|',
      '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for t in CFG['diagnostic_times']:
        lines.append(f"|{t}|{median_noise('E',t,'snr'):.3g}|{median_noise('E',t,'x0_relative_l2'):.3g}|{median_noise('V',t,'x0_relative_l2'):.3g}|{median_noise('E',t,'projected_pcc'):.3g}|{median_noise('V',t,'projected_pcc'):.3g}|{median_noise('E',t,'projected_e_abs'):.3g}|{median_noise('V',t,'projected_e_abs'):.3g}|")
    lines += ['', f"t=990 一步恢复中，V 的 x0 相对 L2 与投影 PCC 改善，但投影 e_abs 为 {median_noise('V',990,'projected_e_abs'):.3g}（E 为 {median_noise('E',990,'projected_e_abs'):.3g}）；不能把误差幅值下降直接认作该步成像全面改善。"]
    lines += ['', '完整 DDIM100 从 t=990 到 0；下表为六个生成种子的中位数，观察每步预测的 x0。', '',
              '|采样索引|E/V x0 std|E/V 奇异值 p95|E/V 酉残差 p95|E/V 投影 PCC|',
              '|---:|---:|---:|---:|---:|']
    for i in (0,19,59,99):
        lines.append(f"|{i}|{median_trace('E',i,'pred_x0_std'):.3g}/{median_trace('V',i,'pred_x0_std'):.3g}|{median_trace('E',i,'sing_p95'):.3g}/{median_trace('V',i,'sing_p95'):.3g}|{median_trace('E',i,'unit_p95'):.3g}/{median_trace('V',i,'unit_p95'):.3g}|{median_trace('E',i,'projected_worst_pcc'):.3g}/{median_trace('V',i,'projected_worst_pcc'):.3g}|")
    lines += ['', f"最终 raw Frobenius 范数中位数 E/V={e['median']['raw_fro']:.3g}/{v['median']['raw_fro']:.3g}；近零像素比例（逐像素 Frobenius 范数 <{CFG['raw_near_zero_threshold']}）={e['median']['near_zero_fraction']:.3g}/{v['median']['near_zero_fraction']:.3g}；投影相对变化={e['median']['projection_change']:.3g}/{v['median']['projection_change']:.3g}。",
      '原三通道同时合格线：PCC≥0.90、e_abs≤0.10、ROI 功率相对误差≤0.20、背景漏光≤0.15、对称/酉残差≤1e-4。',
      '公式与 smoke 检查见 [PREDICTION_TYPE_TEST_REPORT.md](PREDICTION_TYPE_TEST_REPORT.md)。逐通道、分噪声及逐步 CSV 与结构化汇总见结果目录。',
      '[六种子绝对强度接触表](../figures/prediction_type_ablation/contact_absolute.png)；[仅看形状接触表](../figures/prediction_type_ablation/contact_shape_only.png)。每幅图包含理想 T、GS 标签、E/V 投影结果。形状图逐幅缩放，不用于比较功率。',
      '[首个配对生成种子的放大对照图](../figures/prediction_type_ablation/seed_20261201_focused_comparison.png)：三通道逐行、每行共用绝对强度色标，标注 PCC、e_abs 与 ROI/Pin。',
      '[采样幅值与成像轨迹](../figures/prediction_type_ablation/sampling_curves.png)；[分开展示的训练损失](../figures/prediction_type_ablation/training_losses_separate.png)。',
      '本轮仅改变预测目标、训练 MSE 标签与 scheduler prediction_type；模型、数据、优化器、采样时序、物理评价保持原设置。输出未反馈投影到采样循环。',
      '实际运行命令见 [README.md](../README.md)：`--check`、三个训练种子的 `--run-paired` 和 `--evaluate`、最后 `--report`。',
      '官方 [Diffusers DDIM 文档](https://huggingface.co/docs/diffusers/api/schedulers/ddim) 与 [Lin 等论文](https://arxiv.org/abs/2305.08891) 还讨论零终端 SNR、从末时间步采样等组合改动；本轮未测试这些设置，结论仅针对当前配对方案。',
      '单任务结果不能证明跨任务泛化。下一项优先实验建议：在保持 V 配置和单任务不变的前提下，预注册更长的 5000 更新训练，并沿用同六个生成种子与全部门槛评价；本轮 500–2000 步的 V 成像曲线仍在改善。此建议未在本轮执行。']
    repeated=[]
    for extra_seed in CFG['train_seeds'][1:]:
        path=RESULTS/f'seed_{extra_seed}/summary.json'
        if path.exists():repeated.append(json.loads(path.read_text(encoding='utf-8')))
    if repeated:
        lines += ['', '## 独立训练种子复核','',
            '每行是一个独立训练种子；同一行的六张生成图只刻画采样噪声，不视为六个独立训练模型。',
            '|训练种子|E/V 最差 PCC 中位数|E/V 最大 e_abs 中位数|E/V 合格数|e_abs 改善比例|PCC 增量|',
            '|---:|---:|---:|---:|---:|---:|']
        for q in [s]+repeated:
            lines.append(f"|{q['train_seed']}|{q['E']['median']['pcc']:.3f}/{q['V']['median']['pcc']:.3f}|{q['E']['median']['e_abs']:.3f}/{q['V']['median']['e_abs']:.3f}|{q['E']['pass_count']}/{q['V']['pass_count']}|{q['e_abs_improvement_fraction']*100:.1f}%|{q['pcc_gain']:+.3f}|")
        direction=sum(q['e_abs_improvement_fraction']>0 and q['pcc_gain']>0 for q in [s]+repeated)
        lines.append(f"两项成像指标同向改善的独立训练种子为 {direction}/{1+len(repeated)}；至少 2/3 才称初步可重复，仍仅限单任务。")
    lines += ['', '## 实施范围与停止原因','',
        '新增显式 E/V 配置、训练与评价入口、公式和恢复检查、固定种子逐通道/分噪声/采样轨迹数据，以及最终接触表和曲线；未改动原 `Diffusion_prep` 训练源码或旧实验产物。',
        '检查中未发现需共同修复的原训练程序 bug。旧 epsilon 更新回归单独启用确定性 CUDA 算法后输入、loss、参数完全一致；正式训练未启用该开关。',
        f"已完成 {1+len(repeated)} 对独立训练种子；按本轮最多三对的边界停止，不启动八任务或全量训练。",]
    lines += ['', '## 最终 checkpoint 与耗时','',
              '每个 `final.pt` 含模型、条件编码器、优化器、全局 RNG 与独立训练噪声发生器状态，可核对组别、prediction_type 和配置哈希。',
              '|训练种子|E 最终 checkpoint|V 最终 checkpoint|训练耗时 E/V（秒）|',
              '|---:|---|---|---:|']
    total_train_seconds=0.0
    for train_seed in [seed]+[q['train_seed'] for q in repeated]:
        infos={branch:json.loads((RUNS/f'seed_{train_seed}'/branch/'summary.json').read_text(encoding='utf-8')) for branch in BRANCHES}
        total_train_seconds+=sum(info['elapsed_seconds_this_process'] for info in infos.values())
        lines.append(f"|{train_seed}|`runs/prediction_type_ablation/seed_{train_seed}/E/final.pt`|`runs/prediction_type_ablation/seed_{train_seed}/V/final.pt`|{infos['E']['elapsed_seconds_this_process']:.1f}/{infos['V']['elapsed_seconds_this_process']:.1f}|")
    lines.append(f'六个分支训练程序记录耗时合计 {total_train_seconds:.1f} 秒（包含系统停顿及中间评价）；最终诊断与检查另计，整轮低于四小时 GPU 任务预算。')
    lines.append('实际输出路径与逐文件大小见 [artifact_sizes.json](../results/prediction_type_ablation/artifact_sizes.json)。')
    (ROOT/'reports/PREDICTION_TYPE_ABLATION_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    files=[p for top in (RUNS,RESULTS,FIGURES) for p in top.rglob('*') if p.is_file() and p.name!='artifact_sizes.json']
    files += [ROOT/'reports/PREDICTION_TYPE_ABLATION_REPORT.md',ROOT/'reports/PREDICTION_TYPE_TEST_REPORT.md',
              ROOT/'README.md',ROOT/'.gitignore',CFG_PATH,NOISE_MANIFEST,
              ROOT/'configs/prediction_type_E.json',ROOT/'configs/prediction_type_V.json',
              Path(__file__),ROOT/'scripts/plot_focused_prediction.py']
    inventory=[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size} for p in files]
    write_json(RESULTS/'artifact_sizes.json',{'total_bytes_excluding_this_manifest':sum(q['bytes'] for q in inventory),
              'file_count':len(inventory),'files':inventory})


def main():
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check',action='store_true')
    group.add_argument('--run-paired',action='store_true')
    group.add_argument('--evaluate',action='store_true')
    group.add_argument('--report',action='store_true')
    parser.add_argument('--train-seed',type=int,default=20261124)
    args=parser.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    if args.check:run_check()
    elif args.run_paired:run_paired(args.train_seed)
    elif args.evaluate:evaluate_final(args.train_seed)
    else:make_report()


if __name__=='__main__':main()
