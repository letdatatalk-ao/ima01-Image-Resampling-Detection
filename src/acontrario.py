"""B4 - Detection a contrario (Sec. III-B) + validation croisee H/V (Sec. III-D).
Convention rho : (n_patches, n_distances), comme patch_correlations_1d / patch_correlations_2d."""
import numpy as np
from scipy.stats import binom


def tested_distances(n_d,r):
    """Domaine C=[[r+1, N-r-1]] (eq. 18), #C=N-2r-1."""
    return range(r+1,n_d-r)


def count_votes(rho,r,exclude=None):
    """k(d) = nb de patches dont rho(P_i,d) est le max de N_r(P_i,d)=[d-r,d+r] (eq. 17).
    exclude : distances structurelles d* (ex. JPEG kN/8) -> suppression Sec. III-C (voir jpeg_valid_mask)."""
    rho=np.asarray(rho)
    if rho.ndim==3:                      # accepte aussi (npr,npc,nd)
        rho=rho.reshape(-1,rho.shape[-1])
    n_patches,n_d=rho.shape
    k=np.zeros(n_d,dtype=int)
    if not exclude:
        for d in tested_distances(n_d,r):
            k[d]=np.sum(rho[:,d]==np.max(rho[:,d-r:d+r+1],axis=1))
        return k
    C=np.zeros(n_d,bool); C[list(tested_distances(n_d,r))]=True
    win=np.lib.stride_tricks.sliding_window_view
    for x in rho:
        valid=jpeg_valid_mask(x,exclude)
        idx=np.flatnonzero(valid); y=x[idx]
        if len(y)<2*r+1:
            continue
        # N_r(P_i,d) = les r distances VALIDES les plus proches de chaque cote (2r+1 correlations)
        is_max=y[r:len(y)-r]==win(y,2*r+1).max(axis=1)
        d_ok=idx[r:len(y)-r]
        k[d_ok[is_max & C[d_ok]]]+=1
    return k


def jpeg_distances(N,block=8):
    """Distances des pics dus aux blocs JPEG 8x8 : {kN/8, k=1..7} (Sec. III-C)."""
    return sorted({int(round(k*N/block)) for k in range(1,block)})


def jpeg_valid_mask(x,centers):
    """Pour une ligne x=rho(P_i,.), invalide [d-, d+] autour de chaque d* de centers,
    d- et d+ etant le 2e minimum local de x de chaque cote de d* (Sec. III-C, note 3)."""
    n=len(x); valid=np.ones(n,bool)
    m=np.zeros(n,bool); m[1:-1]=(x[1:-1]<x[:-2])&(x[1:-1]<x[2:])
    mins=np.flatnonzero(m)
    for c in centers:
        left=mins[mins<c][::-1]; right=mins[mins>c]
        lo=left[1] if len(left)>1 else 0
        hi=right[1] if len(right)>1 else n-1
        valid[lo:hi+1]=False
    return valid


def binomial_tail(k,n,p):
    """B(k,n,p)=P(X>=k), X~Bin(n,p).  sf(k-1)=P(X>k-1)=P(X>=k)."""
    return binom.sf(k-1,n,p)


def compute_nfa(rho,r,exclude=None):
    """NFA(d)=#C * B(k(d), #P, 1/(2r+1))  (eq. 21). inf hors de C."""
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
    """Distances validees : NFA(d)<eps."""
    return np.where(nfa<eps)[0]


def cross_validate(nfa_1,nfa_2,N1,N2,beta=0.005):
    """eq. 23 : NFA1_valid(d1)=max(NFA1(d1), min_{|x/N2-d1/N1|<beta} NFA2(x)).
    nfa_1 : axe teste (taille N1), nfa_2 : autre axe (taille N2)."""
    x=np.arange(len(nfa_2))
    valid=np.full(len(nfa_1),np.inf)
    for d1 in np.where(np.isfinite(nfa_1))[0]:
        ok=np.abs(x/N2-d1/N1)<beta
        if np.any(ok):
            valid[d1]=max(nfa_1[d1],np.min(nfa_2[ok]))
    return valid


def detect_2d(rho_v,rho_h,H,W,r=20,beta=0.005,eps=1.0,cross_val=True,jpeg=False):
    """rho_v : decalages verticaux (d<H), rho_h : horizontaux (d<W). dmax doit valoir H-1 / W-1.
    jpeg=True : suppression des distances JPEG kN/8 avant le comptage des votes (Sec. III-C)."""
    nfa_v=compute_nfa(rho_v,r,jpeg_distances(H) if jpeg else None)
    nfa_h=compute_nfa(rho_h,r,jpeg_distances(W) if jpeg else None)
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