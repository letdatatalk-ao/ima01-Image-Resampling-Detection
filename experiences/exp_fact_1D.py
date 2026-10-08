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
facteurs=[0.6,0.75,0.9,1.1,1.25,1.5,2.0,2.5]

lognfa_best=[]   # log10 du meilleur (plus bas) NFA sur les vraies distances
detecte=[]       # True si au moins une distance detectee (eps=1)
d_attendu=[]     # d = M mod N (la distance theorique principale)

for q in facteurs:
    N=int(round(q*M))
    dmax=N-1
    u=rng.standard_normal(M)
    v=resample_1d(u,N,kernel_nn)
    rho=patch_correlations_1d(dft(v),L,dmax)
    nfa=compute_nfa(rho,r)
    with np.errstate(divide="ignore"):
        lognfa_best.append(np.log10(np.min(nfa)))
    detecte.append(len(decide(nfa,1.0))>0)
    d_attendu.append(M%N)

facteurs=np.array(facteurs)

# panneaux
fig,ax=plt.subplots(1,2,figsize=(13,5))

# (a) meilleur NFA en fonction du facteur
couleurs=["C2" if ok else "C3" for ok in detecte]
ax[0].scatter(facteurs,lognfa_best,c=couleurs,s=60,zorder=3)
ax[0].plot(facteurs,lognfa_best,color="C0",lw=1,zorder=2)
ax[0].axhline(0,color="C3",ls="--",lw=1,label="seuil log10(eps)=0")
ax[0].axvline(1.0,color="gray",ls=":",lw=1,label="facteur 1 (pas de reech.)")
ax[0].set_xlabel("facteur de reechantillonnage N/M")
ax[0].set_ylabel("log10 meilleur NFA")
ax[0].set_title("(a) Detectabilite vs facteur (vert=detecte, rouge=non)")
ax[0].legend(fontsize=8)

# (b) distance attendue d = M mod N en fonction du facteur
ax[1].plot(facteurs,d_attendu,"o-",color="C1")
ax[1].set_xlabel("facteur de reechantillonnage N/M")
ax[1].set_ylabel("d = M mod N")
ax[1].set_title("(b) Distance de la trace selon le facteur")

fig.suptitle("E4 - Effet du facteur de reechantillonnage (M=100, L=8, r=20, plus proche voisin)",
             fontsize=13)
fig.subplots_adjust(wspace=0.25,top=0.88)
fig.savefig("exp_facteur.png",dpi=130)
print("Figure enregistree : exp_facteur.png")

print("\n N/M |  N  | d=M mod N | log10 best NFA | detecte")
for i,q in enumerate(facteurs):
    N=int(round(q*M))
    print(f"{q:4.2f}|{N:4d} |{d_attendu[i]:9d} | {lognfa_best[i]:14.1f} | {detecte[i]}")