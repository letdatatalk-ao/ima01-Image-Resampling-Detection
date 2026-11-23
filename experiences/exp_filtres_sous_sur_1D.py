import sys,os
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from src.resampling import resample_1d, kernel_nn, linear_kernel, cubic_kernel, lanczos_kernel
from src.dft import dft
from src.spectral import patch_correlations_1d
from src.acontrario import compute_nfa
def signal_colore(M,rng):
    """Signal a spectre en 1/f : energie concentree en basses frequences,
    comme une image naturelle (au lieu du bruit blanc a spectre plat)."""
    w=rng.standard_normal(M)
    W=np.fft.fft(w)
    f=np.fft.fftfreq(M); f[0]=1e-6          
    W=W/np.abs(f)                         
    return np.real(np.fft.ifft(W))

M,L=200,8                                  
rng=np.random.default_rng(0)

u=signal_colore(M,rng)
noyaux={"plus proche voisin":kernel_nn,"lineaire":linear_kernel,
        "cubique":cubic_kernel,"lanczos":lanczos_kernel}
facteurs=np.arange(0.50,2.01,0.05)          
u=rng.standard_normal(M)

best={nom:[] for nom in noyaux}             
for q in facteurs:
    N=int(round(q*M)); dmax=N-1
    r=min(20,max(3,N//6))                   
    for nom,ker in noyaux.items():
        v=resample_1d(u,N,ker)
        nfa=compute_nfa(patch_correlations_1d(dft(v),L,dmax),r)
        with np.errstate(divide="ignore"):
            best[nom].append(np.log10(max(np.min(nfa),1e-300)))

fig,ax=plt.subplots(figsize=(12,6))
for nom in noyaux:
    ax.plot(facteurs,best[nom],"-o",ms=3,label=nom)
ax.axhline(0,color="C3",ls="--",lw=1,label="seuil log10(eps)=0")
ax.axvline(1.0,color="gray",ls=":",lw=1,label="facteur 1 (pas de reech.)")
ax.axvspan(facteurs[0],1.0,color="C0",alpha=0.05)     # zone sous-echantillonnage
ax.text(0.7,ax.get_ylim()[0]*0.9,"SOUS-echantillonnage",color="C0",fontsize=9)
ax.set_xlabel("facteur de reechantillonnage N/M")
ax.set_ylabel("log10 meilleur NFA  (plus bas = plus facile a detecter)")
ax.set_title(f"Point 3 : detectabilite par noyau, sous vs sur-echantillonnage (M={M}, L={L})")
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig("exp_filtres.png",dpi=130)
print("Figure : exp_filtres.png")

print("\nfact |  nn   | lin  | cub  | lanczos")
for i,q in enumerate(facteurs):
    print(f"{q:4.2f} |{best['plus proche voisin'][i]:6.1f} |{best['lineaire'][i]:5.1f} |{best['cubique'][i]:5.1f} |{best['lanczos'][i]:6.1f}")