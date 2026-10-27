import sys,os
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from src.resampling import resample_1d, kernel_nn
from src.dft import dft
from src.spectral import patch_correlations_1d
from src.acontrario import compute_nfa, decide

#Effet du facteur de reechantillonnage N/M sur la detection (signal 1D)
rng=np.random.default_rng(0)
M,L,r=100,8,20
facteurs=np.arange(0.25,10.01,0.25)   

lognfa_best=[]   
detecte=[]      

for q in facteurs:
    N=int(round(q*M))
    if N<2*r+2:          
        lognfa_best.append(np.nan)
        detecte.append(False)
        continue
    dmax=N-1
    u=rng.standard_normal(M)
    v=resample_1d(u,N,kernel_nn)
    rho=patch_correlations_1d(dft(v),L,dmax)
    nfa=compute_nfa(rho,r)
    with np.errstate(divide="ignore"):
        lognfa_best.append(np.log10(np.min(nfa)))
    detecte.append(len(decide(nfa,1.0))>0)

lognfa_best=np.array(lognfa_best)

fig,ax=plt.subplots(figsize=(12,5))
couleurs=["C2" if ok else "C3" for ok in detecte]
ax.scatter(facteurs,lognfa_best,c=couleurs,s=40,zorder=3)
ax.plot(facteurs,lognfa_best,color="C0",lw=1,zorder=2)
ax.axhline(0,color="C3",ls="--",lw=1,label="seuil log10(eps)=0")
ax.axvline(1.0,color="gray",ls=":",lw=1,label="facteur 1 (pas de reech.)")
for q in range(1,11):      # repere les facteurs entiers
    ax.axvline(q,color="lightgray",ls=":",lw=0.6,zorder=1)
ax.set_xlabel("facteur de reechantillonnage N/M")
ax.set_ylabel("log10 meilleur NFA")
ax.set_title("E4 - Detectabilite vs facteur de reechantillonnage (vert=detecte, rouge=non)\n"
             "M=100, L=8, r=20, plus proche voisin")
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig("exp_facteur.png",dpi=130)
print("Figure enregistree : exp_facteur.png")

print("\n N/M |  N   | log10 best NFA | detecte")
for i,q in enumerate(facteurs):
    N=int(round(q*M))
    val="  nan " if np.isnan(lognfa_best[i]) else f"{lognfa_best[i]:6.1f}"
    print(f"{q:4.2f}|{N:5d} | {val}        | {detecte[i]}")