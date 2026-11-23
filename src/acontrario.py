"""Decision a contrario (rapport Sections 2.3, 3.3, 4.3 et 5.3).

rho : tableau (n_patches, n_distances), comme renvoye par patch_correlations_1d / _2d.
    votes      k(d) = nombre de patches dont rho(P_i, d) est le maximum sur [d-r, d+r]     (eq. 6)
    loi H0     k(d) ~ Binomiale(n_patches, p = 1/(2r+1))                                  (eq. 7)
    NFA        NFA(d) = #C * P[B(n, p) >= k(d)],  C = {r+1, ..., N-r-1}                   (eq. 8)
    decision   d est detecte si NFA(d) < eps
    2D         validation croisee entre l'axe vertical et l'axe horizontal                (eq. 11)
    JPEG       les pics des blocs 8x8 (d = kN/8) ne votent pas
"""
import numpy as np
from scipy.stats import binom


def tested_distances(n_d,r):
    """Domaine teste C = {r+1, ..., n_d-r-1} : la fenetre [d-r, d+r] reste dans le spectre. #C = n_d-2r-1."""
    return range(r+1,n_d-r)


def count_votes(rho,r,exclude=None):
    """k(d) pour d dans C. exclude = distances JPEG d* (ou None) : elles et leur voisinage ne votent pas."""
    rho=np.asarray(rho)
    if rho.ndim==3:                                  # (npr, npc, nd) -> (npr*npc, nd)
        rho=rho.reshape(-1,rho.shape[-1])
    n_patches,n_d=rho.shape
    k=np.zeros(n_d,dtype=int)
    if not exclude:
        for d in tested_distances(n_d,r):
            is_max=rho[:,d]==np.max(rho[:,d-r:d+r+1],axis=1)   # un booleen par patch
            k[d]=np.sum(is_max)
        return k
    # JPEG : pour chaque patch, on retire les distances invalides puis on compare
    # chaque distance valide a ses r voisines valides de chaque cote (2r+1 valeurs).
    in_C=np.zeros(n_d,bool); in_C[list(tested_distances(n_d,r))]=True
    windows=np.lib.stride_tricks.sliding_window_view
    for x in rho:
        idx=np.flatnonzero(jpeg_valid_mask(x,exclude))
        y=x[idx]
        if len(y)<2*r+1:
            continue
        is_max=y[r:len(y)-r]==windows(y,2*r+1).max(axis=1)
        d_ok=idx[r:len(y)-r]
        k[d_ok[is_max&in_C[d_ok]]]+=1
    return k


def jpeg_distances(N,block=8):
    """Distances des pics des blocs JPEG : {k N / block, k = 1..block-1}."""
    return sorted({int(round(k*N/block)) for k in range(1,block)})


def jpeg_valid_mask(x,centers):
    """x = rho(P_i, .). Invalide, autour de chaque d* de centers, l'intervalle qui va jusqu'au
    2e minimum local de x de chaque cote (le pic JPEG et ses flancs)."""
    n=len(x)
    valid=np.ones(n,bool)
    is_min=np.zeros(n,bool)
    is_min[1:-1]=(x[1:-1]<x[:-2])&(x[1:-1]<x[2:])
    minima=np.flatnonzero(is_min)
    for c in centers:
        left=minima[minima<c][::-1]; right=minima[minima>c]
        lo=left[1] if len(left)>1 else 0
        hi=right[1] if len(right)>1 else n-1
        valid[lo:hi+1]=False
    return valid


def binomial_tail(k,n,p):
    """P[X >= k] pour X ~ B(n, p). binom.sf(k-1) = P[X > k-1] = P[X >= k]."""
    return binom.sf(k-1,n,p)


def compute_nfa(rho,r,exclude=None):
    """NFA(d) = #C * P[B(n_patches, 1/(2r+1)) >= k(d)] pour d dans C, +inf ailleurs."""
    rho=np.asarray(rho)
    if rho.ndim==3:
        rho=rho.reshape(-1,rho.shape[-1])
    n_patches,n_d=rho.shape
    C=tested_distances(n_d,r)
    if len(C)<=0:
        raise ValueError("pas assez de distances : il faut N > 2r+1")
    k=count_votes(rho,r,exclude)
    nfa=np.full(n_d,np.inf)
    for d in C:
        nfa[d]=len(C)*binomial_tail(k[d],n_patches,1/(2*r+1))
    return nfa


def decide(nfa,eps=1.0):
    """Distances detectees : NFA(d) < eps."""
    return np.where(nfa<eps)[0]


def cross_validate(nfa_1,nfa_2,N1,N2,beta=0.005):
    """NFA_1^valid(d1) = max(NFA_1(d1), min_{|x/N2 - d1/N1| < beta} NFA_2(x)).
    Un pic sur l'axe 1 n'est garde que si l'autre axe a un pic a la meme frequence relative."""
    x=np.arange(len(nfa_2))
    valid=np.full(len(nfa_1),np.inf)
    for d1 in np.where(np.isfinite(nfa_1))[0]:
        same_freq=np.abs(x/N2-d1/N1)<beta
        if np.any(same_freq):
            valid[d1]=max(nfa_1[d1],np.min(nfa_2[same_freq]))
    return valid


def detect_2d(rho_v,rho_h,H,W,r=20,beta=0.005,eps=1.0,cross_val=True,jpeg=False,jpeg_block=8):
    """Votes + NFA sur chaque axe, validation croisee, decision.
    rho_v : decalages verticaux (d = 0..H-1), rho_h : horizontaux (d = 0..W-1)."""
    nfa_v=compute_nfa(rho_v,r,jpeg_distances(H,jpeg_block) if jpeg else None)
    nfa_h=compute_nfa(rho_h,r,jpeg_distances(W,jpeg_block) if jpeg else None)
    if cross_val:
        nfa_v_cv=cross_validate(nfa_v,nfa_h,H,W,beta)
        nfa_h_cv=cross_validate(nfa_h,nfa_v,W,H,beta)
    else:
        nfa_v_cv,nfa_h_cv=nfa_v,nfa_h
    best_v=int(np.argmin(nfa_v_cv)); best_h=int(np.argmin(nfa_h_cv))
    return {"detected":bool(min(nfa_v_cv[best_v],nfa_h_cv[best_h])<eps),
            "nfa_v":nfa_v,"nfa_h":nfa_h,"nfa_v_cv":nfa_v_cv,"nfa_h_cv":nfa_h_cv,
            "d_v":decide(nfa_v_cv,eps),"d_h":decide(nfa_h_cv,eps),
            "best_v":(best_v,float(nfa_v_cv[best_v])),"best_h":(best_h,float(nfa_h_cv[best_h])),
            "min_nfa_nocv":float(min(nfa_v.min(),nfa_h.min()))}
