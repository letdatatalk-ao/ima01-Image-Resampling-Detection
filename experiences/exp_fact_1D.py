"""E4 - Effet du facteur N/M (rapport Section 6.1).
M = 100, N = round(q M) pour q = 0.25 .. 10 (pas 0.25), plus proche voisin, L = 8, r = 20.
Pour chaque facteur : plus petit log10 NFA et decision (au moins une distance avec NFA < 1).
Usage : python experiences/exp_fact_1D.py"""
import sys,os
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.resampling import resample_1d, kernel_nn
from src.dft import dft
from src.spectral import patch_correlations_1d
from src.acontrario import compute_nfa, decide

M,L,r=100,8,20
rng=np.random.default_rng(0)
print(" N/M |   N  | min log10 NFA | detecte")
for q in np.arange(0.25,10.01,0.25):
    N=int(round(q*M))
    if N<2*r+2:                       # il faut au moins une distance testable : N > 2r+1
        continue
    u=rng.standard_normal(M)          # un nouveau signal pour chaque facteur
    nfa=compute_nfa(patch_correlations_1d(dft(resample_1d(u,N,kernel_nn)),L,N-1),r)
    print(f"{q:5.2f}|{N:5d} | {np.log10(nfa.min()):13.1f} | {len(decide(nfa))>0}")
