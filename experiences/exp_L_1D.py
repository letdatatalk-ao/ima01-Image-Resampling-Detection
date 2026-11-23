"""E1 - Effet de la taille de patch L (rapport Section 6.1).
30 bruits blancs M = 100 -> N = 250 (plus proche voisin), r = 20. Pour chaque L :
    bruit de fond = mediane de rho moyen hors des pics (theorie : 1/sqrt(L)),
    hauteur du pic = rho moyen a d = 100, et log10 NFA a d = 100.
On affiche la mediane et l'intervalle interquartile sur les 30 signaux.
Usage : python experiences/exp_L_1D.py"""
import sys,os
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.resampling import resample_1d, kernel_nn
from src.dft import dft
from src.spectral import patch_correlations_1d
from src.acontrario import compute_nfa

M,N,r,N_REAL=100,250,20,30
L_values=[4,8,16,32,50]
d_pic=M%N
hors_pic=np.ones(N,bool); hors_pic[0]=False; hors_pic[95:106]=False; hors_pic[145:156]=False

rng=np.random.default_rng(0)
fond=np.zeros((N_REAL,len(L_values))); pic=np.zeros_like(fond); lognfa=np.zeros_like(fond)
for t in range(N_REAL):
    V=dft(resample_1d(rng.standard_normal(M),N,kernel_nn))
    for i,L in enumerate(L_values):
        rho=patch_correlations_1d(V,L,N-1)
        rho_moy=rho.mean(axis=0)
        fond[t,i]=np.median(rho_moy[hors_pic])
        pic[t,i]=rho_moy[d_pic]
        lognfa[t,i]=np.log10(max(compute_nfa(rho,r)[d_pic],1e-300))

med=lambda a:np.median(a,axis=0)
print(" L | n=N/L | fond (med) | 1/sqrt(L) | pic (med) | log10 NFA(d=100) (med)")
for i,L in enumerate(L_values):
    print(f"{L:3d}| {N//L:5d} | {med(fond)[i]:10.3f} | {1/np.sqrt(L):9.3f} | {med(pic)[i]:9.3f} | {med(lognfa)[i]:8.1f}")
print("L optimal (NFA median le plus bas) :",L_values[int(np.argmin(med(lognfa)))])

fig,ax=plt.subplots(1,2,figsize=(10,3.5))
L=np.array(L_values)
for a,lab in ((fond,"bruit de fond"),(pic,"pic d = M")):
    q1,q3=np.percentile(a,25,axis=0),np.percentile(a,75,axis=0)
    ax[0].errorbar(L,med(a),yerr=[med(a)-q1,q3-med(a)],fmt="o-",capsize=2,label=lab)
ax[0].plot(L,1/np.sqrt(L),"k--",label="1/sqrt(L)"); ax[0].set_xlabel("L"); ax[0].set_ylabel("rho moyen"); ax[0].legend()
q1,q3=np.percentile(lognfa,25,axis=0),np.percentile(lognfa,75,axis=0)
ax[1].errorbar(L,med(lognfa),yerr=[med(lognfa)-q1,q3-med(lognfa)],fmt="o-",capsize=2)
ax[1].axhline(0,color="r",ls="--"); ax[1].set_xlabel("L"); ax[1].set_ylabel("log10 NFA(d = M)")
fig.tight_layout(); out=os.path.join(os.path.dirname(os.path.abspath(__file__)),"exp_L_1D.png"); fig.savefig(out,dpi=120); print("figure ->",out)
