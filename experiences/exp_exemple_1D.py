"""Exemple fil rouge 1D (rapport Section 3).
Bruit blanc u (M = 100) reechantillonne en v (N = 250), plus proche voisin, L = 10, r = 20.
Attendu : pic de correlation et detection a d = M mod N = 100 (et aux harmoniques 50, 150, 200) ;
la reference (bruit blanc de taille N, non reechantillonne) ne doit pas donner de pic.
Usage (depuis code_ima) : python experiences/exp_exemple_1D.py"""
import sys,os
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.resampling import resample_1d, kernel_nn
from src.dft import dft
from src.spectral import pcc_complex, patch_correlations_1d
from src.acontrario import count_votes, compute_nfa, decide

M,N,L,r=100,250,10,20
rng=np.random.default_rng(0)
u=rng.standard_normal(M)
v=resample_1d(u,N,kernel_nn)          # signal reechantillonne
w=rng.standard_normal(N)              # reference : meme taille, pas de reechantillonnage
V,W=dft(v),dft(w)

# 1) le spectre contient une copie de lui-meme, d = M plus loin
print(f"pcc(V[40:50], V[140:150]) = {pcc_complex(V[40:50],V[140:150]):.2f}  (copie, d = M)")
print(f"pcc(V[40:50], V[77:87])   = {pcc_complex(V[40:50],V[77:87]):.2f}  (segment quelconque)")

# 2) correlations des patches, votes, NFA
rho=patch_correlations_1d(V,L,N-1)
rho_ref=patch_correlations_1d(W,L,N-1)
k=count_votes(rho,r)
nfa,nfa_ref=compute_nfa(rho,r),compute_nfa(rho_ref,r)
n=rho.shape[0]
print(f"\n{n} patches, p = 1/(2r+1) = {1/(2*r+1):.3f}, votes attendus par hasard n p = {n/(2*r+1):.2f}")
print(f"votes a d = 100 : {k[100]} / {n}")
with np.errstate(divide="ignore"):
    print("reechantillonne (d, log10 NFA) :",[(int(d),round(float(np.log10(nfa[d])),1)) for d in decide(nfa)])
    print("reference       (d, log10 NFA) :",[(int(d),round(float(np.log10(nfa_ref[d])),1)) for d in decide(nfa_ref)])
