"""E5 - Le rayon r optimal depend de la distance d (rapport Section 6.1).
Bruit blanc M = 100 -> N = 250, noyau lineaire, L = 10. Pour chaque d, r_opt(d) = argmin_r NFA(d).
Usage : python experiences/exp_r_vs_d_1D.py"""
import sys,os
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.resampling import resample_1d, linear_kernel
from src.dft import dft
from src.spectral import patch_correlations_1d
from src.acontrario import compute_nfa

M,N,L=100,250,10
u=np.random.default_rng(0).standard_normal(M)
rho=patch_correlations_1d(dft(resample_1d(u,N,linear_kernel)),L,N-1)   # rho ne depend pas de r

r_values=np.arange(2,N//2,2)
LOG=np.full((len(r_values),N),np.inf)               # LOG[i, d] = log10 NFA(d) pour r = r_values[i]
for i,r in enumerate(r_values):
    with np.errstate(divide="ignore"):
        LOG[i]=np.log10(compute_nfa(rho,r))         # +inf hors du domaine teste

peaks=sorted({(s*k*M)%N for k in range(1,4) for s in (1,-1)})
print("   d  | r_opt | log10 NFA")
for d in peaks:
    i=int(np.argmin(LOG[:,d]))
    print(f" {d:4d} | {r_values[i]:5d} | {LOG[i,d]:7.1f}")
