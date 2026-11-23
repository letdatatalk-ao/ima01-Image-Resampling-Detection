"""E3 - Effet du noyau d'interpolation (rapport Section 6.1).
Un bruit blanc M = 100 -> N = 250 avec chacun des quatre noyaux, L = 8, r = 20 : log10 NFA a d = 100.
Plus le noyau est proche d'un passe-bas ideal, plus les copies du spectre sont faibles.
Usage : python experiences/exp_kernel_1D.py"""
import sys,os
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.resampling import resample_1d, kernel_nn, linear_kernel, cubic_kernel, lanczos_kernel
from src.dft import dft
from src.spectral import patch_correlations_1d
from src.acontrario import compute_nfa

M,N,L,r=100,250,8,20
KERNELS={"plus proche voisin":kernel_nn,"lineaire":linear_kernel,"cubique":cubic_kernel,"lanczos":lanczos_kernel}
u=np.random.default_rng(0).standard_normal(M)     # le meme signal pour les quatre noyaux

print(" noyau               | log10 NFA(d = 100)")
for name,kernel in KERNELS.items():
    nfa=compute_nfa(patch_correlations_1d(dft(resample_1d(u,N,kernel)),L,N-1),r)
    print(f" {name:20s}| {np.log10(nfa[M%N]):6.1f}")
