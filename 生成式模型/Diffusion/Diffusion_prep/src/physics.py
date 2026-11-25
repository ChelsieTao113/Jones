"""Analyzer Fourier propagation and fixed quality metrics."""
import numpy as np
import torch


def numpy_forward(j,p,a):
    near=np.einsum('yxoi,ki->kyxo',j,p,optimize=True)
    far=np.fft.fftshift(np.fft.fft2(near,axes=(1,2),norm='ortho'),axes=(1,2))
    scalar=np.einsum('ko,kyxo->kyx',a.conj(),far,optimize=True)
    return (np.abs(scalar)**2).real


def torch_forward(j,p,a):
    near=torch.einsum('yxoi,ki->kyxo',j,p)
    far=torch.fft.fftshift(torch.fft.fft2(near,dim=(1,2),norm='ortho'),dim=(1,2))
    scalar=torch.einsum('ko,kyxo->kyx',a.conj(),far)
    return scalar.abs().square().real


def quality(image,target,beta=.3):
    result=[]
    roi=image[:,32:96,32:96]
    for actual,desired in zip(roi,target):
        x=np.asarray(actual,dtype=np.float64).ravel()
        y=np.asarray(desired,dtype=np.float64).ravel()
        xc=x-x.mean();yc=y-y.mean()
        denom=np.linalg.norm(xc)*np.linalg.norm(yc)
        eta=float(x.sum()/128**2)
        mask=y>=.05*y.max()
        result.append({'pcc':float(np.dot(xc,yc)/denom) if denom>1e-20 else None,
                       'e_abs':float(np.dot(x-y,x-y)/(np.dot(y,y)+1e-20)),
                       'eta_roi':eta,'eta_relative_error':float(abs(eta-beta)/beta),
                       'black_leakage':float(x[~mask].sum()/x.sum()) if x.sum()>1e-20 else None})
    return result


def residual(j):
    jj=np.swapaxes(j.conj(),-1,-2)@j
    return {'symmetry':float(np.max(abs(j-np.swapaxes(j,-1,-2)))),
            'unitarity':float(np.max(abs(jj-np.eye(2))))}


def accepted(channels,physical,cfg):
    q=cfg['quality']
    return bool(all(v<=q['physical_tolerance'] for v in physical.values()) and all(
        c['pcc'] is not None and c['pcc']>=q['min_pcc'] and
        c['e_abs']<=q['max_e_abs'] and
        c['eta_relative_error']<=q['max_eta_relative_error'] and
        c['black_leakage'] is not None and c['black_leakage']<=q['max_black_leakage']
        for c in channels))
