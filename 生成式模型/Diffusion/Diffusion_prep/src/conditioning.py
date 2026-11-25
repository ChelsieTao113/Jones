"""Three-channel target/polarization/analyzer tokens; labels never enter here."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def polar_real(z):
    return torch.stack((z.real,z.imag),dim=-1).flatten(-2)


class ConditionEncoder(nn.Module):
    def __init__(self, width=128, target_scale=1.2):
        super().__init__()
        self.target_scale=target_scale
        self.patch=nn.Linear(64,width)
        self.polar=nn.Sequential(nn.Linear(8,width),nn.SiLU(),nn.Linear(width,width))
        self.channel=nn.Embedding(3,width)
        # Fixed 2-D sinusoidal positions, not learned from task identity.
        xy=torch.stack(torch.meshgrid(torch.arange(8),torch.arange(8),indexing='ij'),-1).reshape(64,2)
        freq=torch.exp(-math.log(10000)*torch.arange(width//4)/(width//4))
        parts=[]
        for axis in range(2):
            phase=xy[:,axis,None]*freq[None]
            parts.extend((phase.sin(),phase.cos()))
        self.register_buffer('position',torch.cat(parts,-1).float()[None,None])

    def forward(self, target, p, a):
        b=target.shape[0]
        patches=F.unfold((target/self.target_scale).reshape(b*3,1,64,64),8,stride=8)
        patches=patches.transpose(1,2).reshape(b,3,64,64)
        pol=torch.cat((polar_real(p),polar_real(a)),dim=-1)
        tokens=self.patch(patches)+self.polar(pol)[:,:,None,:]
        tokens=tokens+self.position+self.channel(torch.arange(3,device=target.device))[None,:,None,:]
        return tokens.reshape(b,192,-1)
