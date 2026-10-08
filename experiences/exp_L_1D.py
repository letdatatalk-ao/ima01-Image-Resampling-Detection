import numpy as np
import matplotlib.pyplot as plt
from src.resampling import resample_1d, kernel_nn
from src.dft import dft
from src.spectral import patch_correlations_1d
from src.acontrario import compute_nfa

#Effet de la taille de patch L sur la détection (signal 1D)
rng=np.random.default_rng(0)
M,N,r=100,250,20
dmax=N-1
d_pic=M%N
L_values=[2,4,8,16,32,50]
u=rng.standard_normal(M)
v=resample_1d(u,N,kernel_nn)   
w=rng.standard_normal(N)
V,W=dft(v),dft(w)

# grandeurs à mesurer pour chaque L
bruit_fond=[]   # rho moyen hors pic
hauteur_pic=[]   # rho moyen à d=d_pic
n_patches=[]    # nombre de segments = N//L
lognfa_pic=[]   # log10 NFA au pic
lognfa_ref=[]   # log10 du meilleur (plus bas) NFA sur la référence

for L in L_values:
    rho=patch_correlations_1d(V,L,dmax)
    rho_moy=rho.mean(axis=0)
    # bruit de fond
    mask=np.ones(N,dtype=bool)
    mask[0]=False
    mask[max(0,d_pic-5):d_pic+6]=False          
    mask[max(0,(N-d_pic)-5):(N-d_pic)+6]=False  
    bruit_fond.append(np.median(rho_moy[mask]))
    hauteur_pic.append(rho_moy[d_pic])
    n_patches.append(N//L)

    # NFA au pic (signal) et meilleur NFA sur la référence
    nfa=compute_nfa(rho,r)
    rho_ref=patch_correlations_1d(W,L,dmax)
    nfa_ref=compute_nfa(rho_ref,r)
    with np.errstate(divide="ignore"):
        lognfa_pic.append(np.log10(nfa[d_pic]))
        lognfa_ref.append(np.log10(np.min(nfa_ref)))   # le faux positif 

L_values=np.array(L_values)
bruit_fond=np.array(bruit_fond)

# panneaux
fig,ax=plt.subplots(2,2,figsize=(13,8))

# (a) bruit de fond vs L, avec la loi theorique 1/sqrt(L)
ax[0,0].plot(L_values,bruit_fond,"o-",color="C0",label="bruit de fond mesure")
ax[0,0].plot(L_values,bruit_fond[0]*np.sqrt(L_values[0])/np.sqrt(L_values),
              "k--",label="loi theorique 1/sqrt(L)")
ax[0,0].plot(L_values,hauteur_pic,"s-",color="C2",label="hauteur du pic (d=100)")
ax[0,0].set_xlabel("L (taille de patch)")
ax[0,0].set_ylabel("rho moyen")
ax[0,0].set_title("(a) Bruit de fond vs signal")
ax[0,0].legend(fontsize=8)

# (b) nombre de patches vs L (le n de la binomiale)
ax[0,1].plot(L_values,n_patches,"o-",color="C1")
ax[0,1].set_xlabel("L (taille de patch)")
ax[0,1].set_ylabel("nombre de patches  n=N/L")
ax[0,1].set_title("(b) Nombre de votants (n de la binomiale)")

# (c) ecart signal/bruit = hauteur_pic-bruit_fond
ax[1,0].plot(L_values,hauteur_pic-bruit_fond,"o-",color="C3")
ax[1,0].set_xlabel("L (taille de patch)")
ax[1,0].set_ylabel("ecart  (pic-bruit)")
ax[1,0].set_title("(c) Separation signal / bruit de fond")

# (d) log10 NFA au pic (signal) vs reference -> LE compromis
ax[1,1].plot(L_values,lognfa_pic,"o-",color="C0",label="NFA au pic (signal)")
ax[1,1].plot(L_values,lognfa_ref,"s-",color="C7",label="meilleur NFA (reference)")
ax[1,1].axhline(0,color="C3",ls="--",lw=1,label="seuil log10(eps)=0")
ax[1,1].set_xlabel("L (taille de patch)")
ax[1,1].set_ylabel("log10 NFA")
ax[1,1].set_title("(d) Detectabilite : NFA au pic (plus bas = mieux)")
ax[1,1].legend(fontsize=8)

fig.suptitle("E1 - Effet de la taille de patch L (M=100, N=250, r=20, plus proche voisin)",
             fontsize=13)
fig.subplots_adjust(hspace=0.3,wspace=0.25,top=0.92)
fig.savefig("exp_L.png",dpi=130)
print("Figure enregistree : exp_L.png")

print("\n L | n_patches | bruit_fond | pic | log10 NFA pic")
for i,L in enumerate(L_values):
    print(f"{L:3d}|{n_patches[i]:9d} | {bruit_fond[i]:.3f} | {hauteur_pic[i]:.3f} | {lognfa_pic[i]:.1f}")