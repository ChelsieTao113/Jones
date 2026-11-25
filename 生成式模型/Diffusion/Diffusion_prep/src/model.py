"""Local Diffusers conditional U-Net and matched DDPM/DDIM schedulers."""
from diffusers import UNet2DConditionModel, DDPMScheduler, DDIMScheduler
from .conditioning import ConditionEncoder


def build(cfg):
    m=UNet2DConditionModel(**cfg['model'])
    c=ConditionEncoder(cfg['model']['cross_attention_dim'],cfg['target_scale'])
    d=cfg['diffusion']
    ddpm=DDPMScheduler(num_train_timesteps=d['num_train_timesteps'],
                       beta_schedule=d['beta_schedule'],prediction_type=d['prediction_type'],
                       clip_sample=d['clip_sample'])
    ddim=DDIMScheduler.from_config(ddpm.config)
    return m,c,ddpm,ddim
