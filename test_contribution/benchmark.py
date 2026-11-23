"""Comparaison HONNETE : naive / sliding (nous) / integral (papier) / papier-torch.

Pourquoi 4 methodes :
  - naive            : reference lisible (numpy)           -> prouve la justesse
  - sliding (nous)   : FFT + sommes glissantes (numpy)     -> NOTRE methode
  - integral (papier): FFT + images integrales (numpy)     -> l'ALGO du papier, meme backend
  - papier (torch)   : le code officiel tel quel (torch)   -> montre l'effet du BACKEND

Lectures :
  * sliding vs integral  (numpy vs numpy)  = comparaison d'ALGORITHME a backend egal (honnete)
  * integral vs papier   (numpy vs torch)  = effet du BACKEND seul (meme algo)

Equite du chrono :
  * la FFT image->spectre est DANS le chrono pour toutes les methodes (meme perimetre)
  * warm-up (1 appel jete) puis MEDIANE sur plusieurs essais (pas le min, trop optimiste)
  * torch limite a 1 thread (sinon avantage injuste du multi-threading)
"""
import sys, os, time, statistics
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_root, os.path.join(_root, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import spectral as S
from paper_corr_numpy import compute_corr_distances_np

try:
    import torch
    torch.set_num_threads(1)
    from paper_corr import compute_corr_distances as compute_corr_torch
    HAS_TORCH = True
except Exception as e:
    HAS_TORCH = False
    print("torch indisponible -> methode papier-torch sautee (%s)" % type(e).__name__)

def timed(fn, repeats=5, warmup=1):
    for _ in range(warmup):
        fn()
    ts=[]
    for _ in range(repeats):
        s=time.perf_counter(); fn(); ts.append(time.perf_counter()-s)
    return statistics.median(ts)

def run(N, ratio, dmax, do_naive=True, seed=0):
    rng=np.random.default_rng(seed)
    img=rng.standard_normal((N,N))
    h=round(ratio*N); w=round(ratio*N)

    # resultats (une fois) pour la precision, tous ramenes en (npatch, dmax+1)
    spec=np.fft.fft2(img)
    r_naive=S.patch_correlations_naive(spec,(h,w),dmax,axis=0) if do_naive else None
    r_slide=S.patch_correlations_fast(spec,(h,w),dmax,axis=0)
    r_integ=compute_corr_distances_np(img, ratio, dmax).T
    r_torch=compute_corr_torch(img, ratio, dmax).T if HAS_TORCH else None

    ref=r_naive if do_naive else r_slide
    prec=dict(
        slide=np.max(np.abs(r_slide-ref)),
        integ=np.max(np.abs(r_integ-ref)),
    )
    if HAS_TORCH: prec['torch']=np.max(np.abs(r_torch-ref))
    if do_naive:  prec['naive']=0.0

    # temps : FFT image->spectre incluse pour nos methodes (meme perimetre que le papier)
    t=dict()
    if do_naive:
        t['naive']=timed(lambda: S.patch_correlations_naive(np.fft.fft2(img),(h,w),dmax,axis=0), repeats=3)
    t['slide']=timed(lambda: S.patch_correlations_fast(np.fft.fft2(img),(h,w),dmax,axis=0))
    t['integ']=timed(lambda: compute_corr_distances_np(img, ratio, dmax))
    if HAS_TORCH:
        t['torch']=timed(lambda: compute_corr_torch(img, ratio, dmax))

    return dict(N=N, h=h, npatch=r_slide.shape[0], dmax=dmax, prec=prec, t=t)

configs=[(128,0.1,64,True),(256,0.1,128,True),(512,0.1,128,False),(768,0.1,128,False)]
rows=[run(*c) for c in configs]

def col(r, k, fmt):
    return fmt.format(r[k]) if k in r else "   --   "

print("\n==================  PRECISION (erreur max vs reference)  ==================")
print(f"{'N':>5} {'npatch':>7} {'dmax':>5} | {'slide(nous)':>12} {'integral(np)':>13} {'papier(torch)':>14}")
for r in rows:
    p=r['prec']
    pt=f"{p['torch']:.1e}" if 'torch' in p else "   --"
    print(f"{r['N']:>5} {r['npatch']:>7} {r['dmax']:>5} | {p['slide']:>12.1e} {p['integ']:>13.1e} {pt:>14}")

print("\n==================  TEMPS (secondes, mediane)  ==================")
print(f"{'N':>5} {'npatch':>7} {'dmax':>5} | {'naive':>9} {'slide(nous)':>12} {'integral(np)':>13} {'papier(torch)':>14}")
for r in rows:
    t=r['t']
    tn=f"{t['naive']:.3f}" if 'naive' in t else "  --"
    tt=f"{t['torch']:.4f}" if 'torch' in t else "   --"
    print(f"{r['N']:>5} {r['npatch']:>7} {r['dmax']:>5} | {tn:>9} {t['slide']:>12.4f} {t['integ']:>13.4f} {tt:>14}")

print("\n==================  ANALYSE  ==================")
for r in rows:
    t=r['t']
    algo = t['slide']/t['integ']                  # meme backend (numpy) : algo seul
    msg=f"N={r['N']:>4} : slide/integral (numpy, ALGO seul) = {algo:5.2f}x"
    if 'torch' in t:
        backend = t['integ']/t['torch']           # meme algo : backend seul
        msg+=f"  |  integral_np/torch (BACKEND seul) = {backend:5.2f}x"
    print(msg)
print()