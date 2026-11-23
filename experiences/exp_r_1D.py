"""E2 - Effet du rayon de voisinage r (rapport Section 6.1).
Memes 30 signaux que E1, L = 8. Pour chaque r : p = 1/(2r+1), votes k(100) et log10 NFA(100).
Un grand r rend un vote par hasard plus rare, jusqu'a ce que la fenetre [d-r, d+r] contienne
un autre pic (harmonique) : le patch vote alors pour l'autre pic et le pic perd des votes.
Usage : python experiences/exp_r_1D.py"""
import sys,os
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.resampling import resample_1d, kernel_nn
from src.dft import dft
from src.spectral import patch_correlations_1d
from src.acontrario import compute_nfa, count_votes

M,N,L,N_REAL=100,250,8,30
r_values=[2,5,10,20,40,80]
d_pic=M%N

rng=np.random.default_rng(0)
lognfa=np.zeros((N_REAL,len(r_values))); votes=np.zeros_like(lognfa)
for t in range(N_REAL):
    rho=patch_correlations_1d(dft(resample_1d(rng.standard_normal(M),N,kernel_nn)),L,N-1)   # rho ne depend pas de r
    for i,r in enumerate(r_values):
        lognfa[t,i]=np.log10(max(compute_nfa(rho,r)[d_pic],1e-300))
        votes[t,i]=count_votes(rho,r)[d_pic]

med=lambda a:np.median(a,axis=0)
print(" r | p=1/(2r+1) | #C=N-2r-1 | votes k(100) (med) | log10 NFA(100) (med)")
for i,r in enumerate(r_values):
    print(f"{r:3d}| {1/(2*r+1):10.4f} | {N-2*r-1:9d} | {med(votes)[i]:18.0f} | {med(lognfa)[i]:8.1f}")
print("r optimal (NFA median le plus bas) :",r_values[int(np.argmin(med(lognfa)))])

fig,ax=plt.subplots(figsize=(5,3.5))
q1,q3=np.percentile(lognfa,25,axis=0),np.percentile(lognfa,75,axis=0)
ax.errorbar(r_values,med(lognfa),yerr=[med(lognfa)-q1,q3-med(lognfa)],fmt="o-",capsize=2)
ax.set_xscale("log"); ax.axhline(0,color="r",ls="--"); ax.set_xlabel("r"); ax.set_ylabel("log10 NFA(d = M)")
fig.tight_layout(); out=os.path.join(os.path.dirname(os.path.abspath(__file__)),"exp_r_1D.png"); fig.savefig(out,dpi=120); print("figure ->",out)
