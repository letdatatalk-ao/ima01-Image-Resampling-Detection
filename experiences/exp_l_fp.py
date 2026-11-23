import sys,os
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from src.resampling import linear_kernel
from src.dft import dft
from src.spectral import patch_correlations_1d
from src.acontrario import compute_nfa

# Choix de L par les faux positifs : on calibre sur un signal NON reechantillonne (H0)
rng=np.random.default_rng(0)
N,r=250,20
L_values=[4,5,7,8,9,12,16,17]                 # pairs ET impairs, pour comparer
seuils=np.array([10.0**e for e in range(0,-7,-1)])   # eps de 1 a 1e-6
NREAL=50                                 # plusieurs bruits blancs -> moyenne stable

fp_moyen={}                              # fp_moyen[L] = #faux positifs moyen par seuil
for L in L_values:
    acc=np.zeros(len(seuils))
    for _ in range(NREAL):
        w=rng.standard_normal(N)         
        nfa=compute_nfa(patch_correlations_1d(dft(w),L,N-1),r)
        for j,eps in enumerate(seuils):
            acc[j]+=np.sum(nfa<eps)      
    fp_moyen[L]=acc/NREAL

# courbe : #FP (y) en fonction du seuil eps (x), une courbe par L
fig,ax=plt.subplots(figsize=(10,6))
for L in L_values:
    style="-o" if L%2 else "--s"          # trait plein = impair, pointille = pair
    ax.plot(seuils,fp_moyen[L],style,label=f"L={L} ({'impair' if L%2 else 'pair'})")
ax.plot(seuils,seuils,"k:",lw=1.5,label="garantie a contrario  #FP = eps")
ax.set_xscale("log"); ax.set_yscale("log")
ax.invert_xaxis()                         # seuil qui se resserre vers la droite
ax.set_xlabel("seuil de NFA  eps"); ax.set_ylabel("nombre de faux positifs moyen")
ax.set_title(f"Choix de L : faux positifs sur signal NON reechantillonne (N={N}, r={r}, {NREAL} realisations)")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("exp_L_fp.png",dpi=130)
print("Figure : exp_L_fp.png")

print("\n  L   |"+"".join(f" eps=1e{e:<3d}|" for e in range(0,-7,-1)))
for L in L_values:
    print(f"{L:3d}{'I' if L%2 else 'p'} |"+"".join(f"{v:8.2f} |" for v in fp_moyen[L]))