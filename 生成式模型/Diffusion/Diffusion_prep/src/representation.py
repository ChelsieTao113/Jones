"""Six real Jones channels and symmetric-unitary Takagi projection."""
import sys
from pathlib import Path

import numpy as np

VENDOR=Path(__file__).resolve().parents[1]/'vendor'
sys.path.insert(0,str(VENDOR))
from takagi_analytic import takagi_project


def encode(j):
    if np.max(abs(j-np.swapaxes(j,-1,-2)))>1e-4:
        raise ValueError('Jones input is not symmetric')
    x=j[...,0,0];b=j[...,0,1];d=j[...,1,1]
    return np.stack([x.real,x.imag,b.real,b.imag,d.real,d.imag],axis=-3).astype(np.float32)


def decode(x):
    z=np.asarray(x)
    if z.shape[-3]!=6:raise ValueError('six channel axis required')
    z=np.moveaxis(z,-3,-1)
    j=np.empty(z.shape[:-1]+(2,2),np.complex64)
    j[...,0,0]=z[...,0]+1j*z[...,1]
    j[...,0,1]=z[...,2]+1j*z[...,3]
    j[...,1,0]=j[...,0,1]
    j[...,1,1]=z[...,4]+1j*z[...,5]
    return j


def project(j):
    return takagi_project(np.asarray(j))[0]
