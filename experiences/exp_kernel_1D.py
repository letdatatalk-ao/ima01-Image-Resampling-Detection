import sys,os
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from src.resampling import resample_1d, kernel_nn, linear_kernel
from src.dft import dft
from src.spectral import patch_correlations_1d
from src.acontrario import compute_nfa

#Effet du noyau d'interpolation sur la detection (signal 1D)
rng=np.random.default_rng(0)
M,N,L,r=100,250,8,20
dmax=N-1
d_pic=M%N
kernels={"plus proche voisin":kernel_nn,"lineaire":linear_kernel}
u=rng.standard_normal(M)

# panneaux : une courbe NFA(d) par noyau + un bar chart du NFA au pic
fig,ax=plt.subplots(1,2,figsize=(13,5))

nfa_au_pic={}
d=np.arange(N)
for i,(nom,ker) in enumerate(kernels.items()):
    v=resample_1d(u,N,ker)
    rho=patch_correlations_1d(dft(v),L,dmax)
    nfa=compute_nfa(rho,r)
    nfa_au_pic[nom]=nfa[d_pic]
    with np.errstate(divide="ignore"):
        log_nfa=np.log10(nfa)
    ax[0].plot(d,log_nfa,lw=1,label=nom)

ax[0].axhline(0,color="C3",ls="--",lw=1,label="seuil log10(eps)=0")
ax[0].axvline(d_pic,color="green",ls=":",lw=1)
ax[0].set_xlabel("distance d")
ax[0].set_ylabel("log10 NFA(d)")
ax[0].set_title("(a) Courbe NFA selon le noyau")
ax[0].legend(fontsize=8)

# (b) NFA au pic par noyau (plus bas = detection plus forte)
noms=list(nfa_au_pic.keys())
with np.errstate(divide="ignore"):
    vals=[np.log10(nfa_au_pic[n]) for n in noms]
ax[1].bar(noms,vals,color=["C0","C1"])
ax[1].set_ylabel("log10 NFA au pic (d=100)")
ax[1].set_title("(b) Force de detection par noyau")

fig.suptitle("E3 - Effet du noyau d'interpolation (M=100, N=250, L=8, r=20)",fontsize=13)
fig.subplots_adjust(wspace=0.25,top=0.88)
fig.savefig("exp_kernel.png",dpi=130)
print("Figure enregistree : exp_kernel.png")

print("\n noyau | log10 NFA au pic")
for n in noms:
    with np.errstate(divide="ignore"):
        print(f"{n:20s} | {np.log10(nfa_au_pic[n]):.1f}")