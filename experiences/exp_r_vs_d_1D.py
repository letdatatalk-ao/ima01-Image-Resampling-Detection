import sys,os
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from src.resampling import resample_1d, linear_kernel
from src.dft import dft
from src.spectral import patch_correlations_1d
from src.acontrario import compute_nfa

# r optimal en fonction de la distance d (signal 1D)
rng=np.random.default_rng(0)
M,FAC,L=100,2.5,10
N=int(round(FAC*M)); dmax=N-1
u=rng.standard_normal(M)
v=resample_1d(u,N,linear_kernel)
rho=patch_correlations_1d(dft(v),L,dmax)       # rho ne depend PAS de r : calcule une fois

r_values=np.arange(2,N//2,2)                    # rayons testes
LOG=np.full((len(r_values),N),np.nan)           # log10 NFA[r, d]
for ri,r in enumerate(r_values):
    nfa=compute_nfa(rho,r)
    with np.errstate(divide="ignore"):
        LOG[ri]=np.log10(nfa)                   # inf = hors zone testable

# r optimal pour chaque d : le r qui minimise log NFA (colonne par colonne)
r_opt=np.full(N,np.nan); best=np.full(N,np.nan)
for d in range(N):
    col=LOG[:,d]
    if np.all(~np.isfinite(col)): continue
    ri=np.argmin(np.where(np.isfinite(col),col,np.inf))
    if np.isfinite(col[ri]): r_opt[d]=r_values[ri]; best[d]=col[ri]

LOGd=np.where(np.isfinite(LOG),LOG,np.nan)       # pour l'affichage
dd=np.arange(N)
peaks=sorted(set([(k*M)%N for k in range(1,6)]+[(-k*M)%N for k in range(1,6)]))
peaks=[p for p in peaks if 0<p<N and np.isfinite(r_opt[p])]

fig,ax=plt.subplots(1,2,figsize=(14,5.5))

# (a) carte log NFA(d, r) + crete r_opt(d) + borne geometrique
im=ax[0].imshow(LOGd,origin="lower",aspect="auto",cmap="viridis",
   extent=[0,N,r_values[0],r_values[-1]],vmin=-30,vmax=5)
ax[0].plot(dd,r_opt,color="red",lw=1.5,label="r optimal(d)")
ax[0].plot(dd,np.minimum(dd,N-1-dd),"w--",lw=1,label="borne geometrique min(d, N-1-d)")
ax[0].set_xlabel("distance d"); ax[0].set_ylabel("rayon r")
ax[0].set_title("log10 NFA(d, r) — fonce = mieux (NFA faible)\nle r optimal depend de d")
ax[0].legend(fontsize=8,loc="upper right")
fig.colorbar(im,ax=ax[0],label="log10 NFA")

# (b) r_opt en fonction de d, mis en avant aux harmoniques
ax[1].plot(dd,r_opt,color="red",lw=1,alpha=.35,label="r optimal (tous d)")
ax[1].scatter(peaks,[r_opt[p] for p in peaks],color="C0",s=60,zorder=3,label="aux harmoniques k*M mod N")
ax[1].plot(dd,np.minimum(dd,N-1-dd),"k--",lw=1,label="borne min(d, N-1-d)")
ax[1].set_xlabel("distance d"); ax[1].set_ylabel("r optimal")
ax[1].set_title("r optimal en fonction de d"); ax[1].legend(fontsize=8)

fig.suptitle(f"r optimal depend de d  (M={M}, N={N}, L={L}, noyau lineaire)",fontsize=13)
fig.tight_layout(); fig.savefig("exp_r_vs_d.png",dpi=130)
print("Figure: exp_r_vs_d.png")
print("\n d(harm) | r_opt | log10 NFA | borne min(d,N-1-d)")
for p in peaks:
    print(f"{p:6d}  |{r_opt[p]:5.0f}  |{best[p]:7.1f}   | {min(p,N-1-p)}")