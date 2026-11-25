"""Bounded real-data training and pure-noise DDIM evaluation."""
import hashlib, json, random, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .dataset import ROOT, cfg, fixed_pool, load
from .representation import encode, decode, project
from .physics import numpy_forward, quality, residual, accepted
from .model import build


def write(name,obj):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=float),encoding='utf-8')


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def load_items(records,device):
    items=[]
    for r in records:
        z=load(r)
        items.append({'id':r['task_id'],'x':torch.from_numpy(encode(z['jones'])).to(device),
                      't':torch.from_numpy(z['target_intensity']).to(device),
                      'p':torch.from_numpy(z['polarization']).to(device),
                      'a':torch.from_numpy(z['analyzer']).to(device),
                      'z':z})
    return items


def batch(items,indices):
    subset=[items[i] for i in indices]
    return tuple(torch.stack([q[k] for q in subset]) for k in ('x','t','p','a'))


def save_checkpoint(stage,step,m,c,opt,cc):
    rng={'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state()}
    if torch.cuda.is_available():rng['cuda']=torch.cuda.get_rng_state_all()
    confighash=hashlib.sha256((ROOT/'configs'/'prep_v1.json').read_bytes()).hexdigest()
    state={'model':m.state_dict(),'condition':c.state_dict(),'optimizer':opt.state_dict(),
           'step':step,'rng':rng,'config_sha256':confighash,'stage':stage}
    p=ROOT/'checkpoints'/f'{stage}_{step:05d}.pt';tmp=p.with_suffix('.tmp')
    torch.save(state,tmp);tmp.replace(p)
    # Reload into independently constructed modules; eval outputs must match.
    m2,c2,_,_=build(cc);m2.to(next(m.parameters()).device);c2.to(next(c.parameters()).device)
    state2=torch.load(p,map_location=next(m.parameters()).device,weights_only=False)
    m2.load_state_dict(state2['model']);c2.load_state_dict(state2['condition'])
    m.eval();c.eval();m2.eval();c2.eval()
    with torch.no_grad():
        test=torch.zeros(1,6,128,128,device=next(m.parameters()).device)
        cond=torch.zeros(1,3,64,64,device=test.device)
        pol=torch.zeros(1,3,2,dtype=torch.complex64,device=test.device);pol[...,0]=1
        v=m(test,torch.tensor([999],device=test.device),encoder_hidden_states=c(cond,pol,pol)).sample
        w=m2(test,torch.tensor([999],device=test.device),encoder_hidden_states=c2(cond,pol,pol)).sample
        diff=(v-w).abs().max().item()
    del m2,c2,state2
    m.train();c.train()
    if diff>1e-6:raise RuntimeError(f'checkpoint reload mismatch {diff}')
    return str(p.relative_to(ROOT)),diff


def sample(m,c,ddim,item,seed,cc,condition=None):
    device=next(m.parameters()).device
    gen=torch.Generator(device=device).manual_seed(seed)
    x=torch.randn((1,6,128,128),device=device,generator=gen)
    t,p,a=condition if condition is not None else (item['t'][None],item['p'][None],item['a'][None])
    tokens=c(t,p,a)
    ddim.set_timesteps(cc['diffusion']['inference_steps'],device=device)
    start=time.monotonic();m.eval();c.eval()
    with torch.no_grad():
        for step in ddim.timesteps:
            pred=m(x,step,encoder_hidden_states=tokens).sample
            x=ddim.step(pred,step,x,eta=cc['diffusion']['eta']).prev_sample
    elapsed=time.monotonic()-start
    arr=x[0].float().cpu().numpy()
    raw=decode(arr);phys=project(raw)
    z=item['z'];target=z['target_intensity'];pp=z['polarization'];aa=z['analyzer']
    ir=numpy_forward(raw,pp,aa);ip=numpy_forward(phys,pp,aa)
    return {'x':arr,'raw':raw,'phys':phys,'ir':ir,'ip':ip,'time':elapsed,
            'condition_target':t[0].detach().cpu().numpy(),
            'condition_p':p[0].detach().cpu().numpy(),
            'condition_a':a[0].detach().cpu().numpy(),
            'raw_metrics':quality(ir,target),'physical_metrics':quality(ip,target),
            'raw_residual':residual(raw),'physical_residual':residual(phys),
            'pass':accepted(quality(ip,target),residual(phys),cc)}


def save_sample(stage,update,item,seed,result,cc,tag='correct'):
    stem=f'{stage}_{update:05d}_{item["id"]}_{seed}_{tag}'
    np.savez_compressed(ROOT/'results'/f'{stem}.npz',x0=result['x'],J_raw=result['raw'],
        J_phys=result['phys'],input_polarization=item['z']['polarization'],
        output_analyzer=item['z']['analyzer'],target=item['z']['target_intensity'],
        condition_target=result['condition_target'],condition_polarization=result['condition_p'],
        condition_analyzer=result['condition_a'],
        gs_label=item['z']['jones'],raw_reconstruction=result['ir'],
        physical_reconstruction=result['ip'],seed=np.int64(seed),
        inference_steps=np.int32(cc['diffusion']['inference_steps']),
        scheduler_config=np.array(json.dumps(cc['diffusion'])),elapsed_seconds=result['time'])
    # Independent disk audit by explicit per-channel formula.
    with np.load(ROOT/'results'/f'{stem}.npz',allow_pickle=False) as disk:
        j=disk['J_phys'];p=disk['input_polarization'];a=disk['output_analyzer'];observed=disk['physical_reconstruction']
        independent=np.empty_like(observed,dtype=np.float64)
        for k in range(3):
            ex=j[:,:,0,0]*p[k,0]+j[:,:,0,1]*p[k,1]
            ey=j[:,:,1,0]*p[k,0]+j[:,:,1,1]*p[k,1]
            ux=np.fft.fftshift(np.fft.fft2(ex,norm='ortho'))
            uy=np.fft.fftshift(np.fft.fft2(ey,norm='ortho'))
            independent[k]=abs(a[k,0].conjugate()*ux+a[k,1].conjugate()*uy)**2
        audit=float(np.linalg.norm(independent-observed)/np.linalg.norm(independent))
    return {'task_id':item['id'],'seed':seed,'tag':tag,'update':update,'file':f'results/{stem}.npz',
            'elapsed_seconds':result['time'],'raw_metrics':result['raw_metrics'],
            'physical_metrics':result['physical_metrics'],'raw_residual':result['raw_residual'],
            'physical_residual':result['physical_residual'],'pass':result['pass'],
            'disk_audit_relative_l2':audit}


def train_step(m,c,ddpm,opt,items,cc,update):
    accum=cc['optimizer']['effective_batch'];opt.zero_grad(set_to_none=True)
    losses=[]
    for micro in range(accum):
        index=(update*accum+micro)%len(items)
        x,t,p,a=batch(items,[index]);noise=torch.randn_like(x)
        ts=torch.randint(0,ddpm.config.num_train_timesteps,(1,),device=x.device)
        noisy=ddpm.add_noise(x,noise,ts)
        pred=m(noisy,ts,encoder_hidden_states=c(t,p,a)).sample
        loss=F.mse_loss(pred,noise)
        if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
        (loss/accum).backward();losses.append(loss.item())
    grad_m=float(torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in m.parameters() if p.grad is not None])).item())
    grad_c=float(torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in c.parameters() if p.grad is not None])).item())
    torch.nn.utils.clip_grad_norm_(list(m.parameters())+list(c.parameters()),cc['optimizer']['grad_clip'])
    opt.step()
    return float(np.mean(losses)),grad_m,grad_c


def run(stage,resume=None):
    cc=cfg();seed_all(cc['seed']);device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cpu':raise RuntimeError('GPU required for this full-size training check')
    records=fixed_pool()[:1 if stage in ('smoke','single') else cc['tiny_count']]
    items=load_items(records,device)
    m,c,ddpm,ddim=build(cc);m.to(device);c.to(device)
    m.enable_gradient_checkpointing()
    opt=torch.optim.AdamW(list(m.parameters())+list(c.parameters()),
                          lr=cc['optimizer']['learning_rate'],weight_decay=cc['optimizer']['weight_decay'])
    start_update=0
    if resume is not None:
        if stage not in ('single','eight'):raise ValueError('resume only supports tiny training')
        state=torch.load(resume,map_location='cpu',weights_only=False)
        expected=hashlib.sha256((ROOT/'configs'/'prep_v1.json').read_bytes()).hexdigest()
        if state['stage']!=stage or state['config_sha256']!=expected:raise ValueError('checkpoint mismatch')
        m.load_state_dict(state['model']);c.load_state_dict(state['condition']);opt.load_state_dict(state['optimizer'])
        random.setstate(state['rng']['python']);np.random.set_state(state['rng']['numpy'])
        torch.set_rng_state(state['rng']['torch']);torch.cuda.set_rng_state_all(state['rng']['cuda'])
        start_update=state['step']
    parameter_count=sum(x.numel() for x in m.parameters());condition_count=sum(x.numel() for x in c.parameters())
    if stage=='evaluate':
        for training_stage in ('single','eight'):
            files=sorted((ROOT/'checkpoints').glob(f'{training_stage}_*.pt'))
            if files:
                state=torch.load(files[-1],map_location=device,weights_only=False)
                m.load_state_dict(state['model']);c.load_state_dict(state['condition'])
                chosen=items if training_stage=='eight' else items[:1]
                evaluate(training_stage,state['step'],m,c,ddim,chosen,cc,final=True)
        return
    max_updates=20 if stage=='smoke' else cc['budget'][f'{stage}_max_updates']
    if stage!='smoke':
        # Explicit budget decision based on the saved 20-update measurement.
        smoke=json.loads((ROOT/'results'/'smoke.json').read_text(encoding='utf-8'))
        sec=smoke['seconds_per_update'];sample_sec=smoke['sample_seconds_estimate']
        reserved=(len(items)*3+4)*sample_sec+600
        prior=0
        if stage=='eight':
            prior_file=ROOT/'results'/'single_summary.json'
            if prior_file.exists():prior=json.loads(prior_file.read_text(encoding='utf-8'))['elapsed_seconds']
        remaining=max(0,cc['budget']['wall_seconds']-smoke['elapsed_seconds']-prior-reserved)
        max_updates=min(max_updates,max(1,int(remaining/sec/(2 if stage=='single' else 1))))
        cc['budget'][f'{stage}_planned_updates']=max_updates
        write('results/budget_plan.json',{'smoke_seconds_per_update':sec,
              'sample_seconds_estimate':sample_sec,'planned_single':cc['budget'].get('single_planned_updates'),
              'planned_eight':cc['budget'].get('eight_planned_updates'),
              'rule':'half remaining estimate to single; remainder to eight; 3h global cap'})
    if stage=='smoke':
        x,t,p,a=batch(items,[0]);alter=fixed_pool()[1];other=load_items([alter],device)[0]
        with torch.no_grad():
            token=c(t,p,a);changed_p=c(t,other['p'][None],other['a'][None]);changed_t=c(other['t'][None],p,a)
            token_diff={'polarization':float((token-changed_p).abs().max()),
                        'target':float((token-changed_t).abs().max())}
        before_m=next(m.parameters()).detach().clone();before_c=next(c.parameters()).detach().clone()
    if stage=='single' and start_update==0:evaluate(stage,0,m,c,ddim,items,cc,final=False)
    start=time.monotonic();trace=[];torch.cuda.reset_peak_memory_stats()
    old_trace=ROOT/'results'/f'{stage}_trace.json'
    if start_update and old_trace.exists():
        trace=[r for r in json.loads(old_trace.read_text(encoding='utf-8')) if r['update']<=start_update]
    for update in range(start_update+1,max_updates+1):
        loss,gm,gc=train_step(m,c,ddpm,opt,items,cc,update-1)
        trace.append({'update':update,'loss':loss,'grad_unet':gm,'grad_condition':gc,
                      'elapsed_seconds':time.monotonic()-start})
        if update==1 or update%10==0:print(stage,update,loss,round(trace[-1]['elapsed_seconds'],1),flush=True)
        if stage!='smoke' and (update%cc['budget']['eval_every']==0 or update==max_updates):
            save_checkpoint(stage,update,m,c,opt,cc)
            evaluate(stage,update,m,c,ddim,items[:1],cc,final=False)
        if stage!='smoke' and time.monotonic()-start>=cc['budget']['wall_seconds']:
            break
    elapsed=time.monotonic()-start;peak=torch.cuda.max_memory_allocated()/2**30
    write(f'results/{stage}_trace.json',trace)
    if stage=='smoke':
        delta_m=float((next(m.parameters())-before_m).abs().max());delta_c=float((next(c.parameters())-before_c).abs().max())
        ck,ck_diff=save_checkpoint(stage,20,m,c,opt,cc)
        # Time one full DDIM100 generation on a real condition.
        res=sample(m,c,ddim,items[0],cc['seed'],cc)
        rec=save_sample(stage,20,items[0],cc['seed'],res,cc)
        summary={'updates':20,'elapsed_seconds':elapsed,'seconds_per_update':elapsed/20,
                 'peak_gib':peak,'parameters_unet':parameter_count,'parameters_condition':condition_count,
                 'parameter_change_unet':delta_m,'parameter_change_condition':delta_c,
                 'token_change':token_diff,'checkpoint':ck,'reload_max_abs':ck_diff,
                 'sample_seconds_estimate':res['time'],'sample':rec}
        write('results/smoke.json',summary)
        md=ROOT/'reports'/'MODEL_SMOKE_REPORT.md'
        md.write_text(f'''# 真实训练步检查\n\n固定配置：Diffusers `UNet2DConditionModel`，6 通道 128²，32/64/128，最低尺度及中间层交叉注意力；条件 `[B,192,128]`，DDPM 1000、epsilon、squaredcos，AdamW 1e-4、有效 batch 4（4 次梯度累积），float32 与梯度检查点。U-Net 参数 {parameter_count:,}，条件编码器 {condition_count:,}。\n\n20 次 optimizer updates 共 {elapsed:.1f} 秒，{elapsed/20:.2f} 秒/次；峰值分配显存 {peak:.2f} GiB。U-Net/条件编码器参数最大变化 {delta_m:.3g}/{delta_c:.3g}，token 对偏振和目标的响应 {token_diff}，checkpoint 重载输出最大差 {ck_diff:.3g}。各次损失、梯度见 [smoke_trace.json](../results/smoke_trace.json)。一次纯噪声 DDIM100 耗时 {res['time']:.1f} 秒。训练链路 {'通过' if min(delta_m,delta_c,token_diff['polarization'],token_diff['target'])>0 else '未通过'}；20 步不代表学会成像。\n''',encoding='utf-8')
    else:
        final_update=trace[-1]['update']
        write(f'results/{stage}_summary.json',{'actual_updates':final_update,'planned_updates':max_updates,
              'elapsed_seconds':elapsed,'peak_gib':peak,'last_loss':trace[-1]['loss']})
        evaluate(stage,final_update,m,c,ddim,items,cc,final=True)


def evaluate(stage,update,m,c,ddim,items,cc,final):
    seeds=cc['budget']['final_seeds'] if final else cc['budget']['final_seeds'][:1]
    records=[]
    for item in items:
        for seed in seeds:
            result=sample(m,c,ddim,item,seed,cc)
            records.append(save_sample(stage,update,item,seed,result,cc))
    if final and len(items)>=2:
        item=items[0];other=items[1];base_seed=seeds[0]
        combinations={
            'whole_other':(other['t'][None],other['p'][None],other['a'][None]),
            'other_polar':(item['t'][None],other['p'][None],other['a'][None]),
            'other_target':(other['t'][None],item['p'][None],item['a'][None])}
        for tag,condition in combinations.items():
            r=sample(m,c,ddim,item,base_seed,cc,condition)
            records.append(save_sample(stage,update,item,base_seed,r,cc,tag))
    write(f'results/{stage}_{update:05d}_evaluation.json',records)
    return records


def resume_check():
    """Load saved model, optimizer and RNG, then perform one genuine update."""
    cc=cfg();device='cuda';seed_all(cc['seed'])
    state=torch.load(ROOT/'checkpoints'/'smoke_00020.pt',map_location='cpu',weights_only=False)
    expected=hashlib.sha256((ROOT/'configs'/'prep_v1.json').read_bytes()).hexdigest()
    if state['config_sha256']!=expected:raise RuntimeError('checkpoint config changed')
    m,c,ddpm,_=build(cc);m.to(device);c.to(device);m.enable_gradient_checkpointing()
    opt=torch.optim.AdamW(list(m.parameters())+list(c.parameters()),
                          lr=cc['optimizer']['learning_rate'],weight_decay=cc['optimizer']['weight_decay'])
    m.load_state_dict(state['model']);c.load_state_dict(state['condition']);opt.load_state_dict(state['optimizer'])
    random.setstate(state['rng']['python']);np.random.set_state(state['rng']['numpy'])
    torch.set_rng_state(state['rng']['torch']);torch.cuda.set_rng_state_all(state['rng']['cuda'])
    items=load_items(fixed_pool()[:1],device)
    first=next(m.parameters()).detach().clone()
    loss,gm,gc=train_step(m,c,ddpm,opt,items,cc,state['step'])
    delta=float((next(m.parameters()).detach()-first).abs().max())
    result={'loaded_step':state['step'],'continued_step':state['step']+1,'loss':loss,
            'grad_unet':gm,'grad_condition':gc,'parameter_change':delta,
            'optimizer_state_entries':len(opt.state)}
    write('results/resume_check.json',result)
    if not np.isfinite(loss) or delta<=0 or len(opt.state)==0:raise RuntimeError('resume failed')
