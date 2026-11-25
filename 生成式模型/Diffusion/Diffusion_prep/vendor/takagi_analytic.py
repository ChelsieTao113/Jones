"""Nearest symmetric unitary for a batch of complex 2x2 matrices.

For symmetric S, its polar factor S(S^H S)^(-1/2) equals the Takagi
projection. The closed 2x2 Hermitian square root avoids a batched 4x4 eigh.
Rank-deficient matrices use the original eigensolver to choose a basis.
"""
import numpy as np
from gs_dataset import takagi_project as eigen_takagi_project


def takagi_project(z, singular_floor=1e-6):
    z64=z.astype(np.complex128)
    s=.5*(z64+np.swapaxes(z64,-1,-2))
    a,b,d=s[...,0,0],s[...,0,1],s[...,1,1]
    h11=abs(a)**2+abs(b)**2
    h22=abs(b)**2+abs(d)**2
    h12=a.conj()*b+b.conj()*d
    sigma=abs(a*d-b*b)
    q=np.sqrt(h11+h22+2*sigma)
    determinant=(h11+sigma)*(h22+sigma)-abs(h12)**2
    safe=(sigma>singular_floor)&(determinant>singular_floor)
    c=np.zeros_like(q)
    np.divide(q,determinant,out=c,where=safe)
    out=np.empty_like(s)
    out[...,0,0]=c*(a*(h22+sigma)-b*h12.conj())
    out[...,0,1]=c*(-a*h12+b*(h11+sigma))
    out[...,1,0]=c*(b*(h22+sigma)-d*h12.conj())
    out[...,1,1]=c*(-b*h12+d*(h11+sigma))
    if np.any(~safe):
        out[~safe]=eigen_takagi_project(z[~safe])[0]
    return out.astype(z.dtype,copy=False),None,None
