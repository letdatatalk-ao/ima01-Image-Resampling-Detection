import sys,os
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from src.resampling import resample_1d, kernel_nn
from src.dft import dft
from src.spectral import patch_correlations_1d
from src.acontrario import compute_nfa, count_votes

#Effet du rayon de voisinage r sur la detection (signal 1D)
rng=np.random.default_rng(0)
M,N,L=100,250,8
dmax=N-1
d_pic=M%N
r_values=[2,5,10,20,40,80]
u=rng.standard_normal(M)
v=resample_1d(u,N,kernel_nn)
w=rng.standard_normal(N)
V,W=dft(v),dft(w)

# rho ne depend PAS de r -> on le calcule une seule fois
rho=patch_correlations_1d(V,L,dmax)
rho_ref=patch_correlations_1d(W,L,dmax)
n_patches=N//L

# grandeurs a mesurer pour chaque r
p_values=[]      # p = 1/(2r+1)
k_pic=[]         # nombre de votes au pic
n_testables=[]   # nombre de distances testees = N-2r
lognfa_pic=[]    # log10 NFA au pic (signal)
lognfa_ref=[]    # log10 meilleur NFA (reference)

for r in r_values:
    p_values.append(1/(2*r+1))
    k=count_votes(rho,r)
    k_pic.append(int(k[d_pic]))
    n_testables.append(N-2*r)
    nfa=compute_nfa(rho,r)
    nfa_r=compute_nfa(rho_ref,r)
    with np.errstate(divide="ignore"):
        lognfa_pic.append(np.log10(nfa[d_pic]))
        lognfa_ref.append(np.log10(np.min(nfa_r)))

r_values=np.array(r_values)

# panneaux
fig,ax=plt.subplots(2,2,figsize=(13,8))

# (a) p = 1/(2r+1) en fonction de r
ax[0,0].plot(r_values,p_values,"o-",color="C0")
ax[0,0].set_xlabel("r (rayon de voisinage)")
ax[0,0].set_ylabel("p = 1/(2r+1)")
ax[0,0].set_title("(a) Proba de vote par hasard")

# (b) nombre de votes au pic vs nombre de distances testables
ax[0,1].plot(r_values,k_pic,"o-",color="C2",label="votes au pic k(100)")
ax[0,1].plot(r_values,n_testables,"s-",color="C1",label="distances testables N-2r")
ax[0,1].set_xlabel("r (rayon de voisinage)")
ax[0,1].set_ylabel("nombre")
ax[0,1].set_title("(b) Votes au pic & distances testees")
ax[0,1].legend(fontsize=8)

# (c) esperance de votes par hasard n*p
ax[1,0].plot(r_values,n_patches*np.array(p_values),"o-",color="C3")
ax[1,0].set_xlabel("r (rayon de voisinage)")
ax[1,0].set_ylabel("n*p (votes attendus par hasard)")
ax[1,0].set_title("(c) Votes attendus sous H0")

# (d) log10 NFA au pic (signal) vs reference
ax[1,1].plot(r_values,lognfa_pic,"o-",color="C0",label="NFA au pic (signal)")
ax[1,1].plot(r_values,lognfa_ref,"s-",color="C7",label="meilleur NFA (reference)")
ax[1,1].axhline(0,color="C3",ls="--",lw=1,label="seuil log10(eps)=0")
ax[1,1].set_xlabel("r (rayon de voisinage)")
ax[1,1].set_ylabel("log10 NFA")
ax[1,1].set_title("(d) Detectabilite : NFA au pic (plus bas = mieux)")
ax[1,1].legend(fontsize=8)

fig.suptitle("E2 - Effet du rayon de voisinage r (M=100, N=250, L=8, plus proche voisin)",
             fontsize=13)
fig.subplots_adjust(hspace=0.3,wspace=0.25,top=0.92)
fig.savefig("exp_r.png",dpi=130)
print("Figure enregistree : exp_r.png")

print("\n r | p=1/(2r+1) | k_pic | N-2r | log10 NFA pic")
for i,r in enumerate(r_values):
    print(f"{r:3d}| {p_values[i]:.4f}   |{k_pic[i]:5d} |{n_testables[i]:5d} | {lognfa_pic[i]:.1f}")