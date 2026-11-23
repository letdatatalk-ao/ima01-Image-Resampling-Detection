"""EXP 1 - Pipeline complet sur bruit blanc 2D synthetique.
bruit blanc MxM -> (reechantillonnage NxN ou non) -> rank transform -> FFT 2D
-> correlations de patches (V et H) -> NFA -> validation croisee -> decision.
Attendu : pics a d = M mod N et d = -M mod N sur les deux axes ; rien sur l'original."""
import sys,os,time
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.resampling import resample_2d_sep,kernel_nn,linear_kernel,cubic_kernel
from src.residual import rank_transform_2d
from src.spectral import patch_correlations_2d
from src.acontrario import detect_2d

R_NEIGH,BETA,EPS=20,0.005,1.0


def pipeline(img):
    """image -> dict de detect_2d"""
    R=rank_transform_2d(img,s=3)
    H,W=R.shape
    rv,rh,_=patch_correlations_2d(np.fft.fft2(R),patch_ratio=10)
    return detect_2d(rv,rh,H,W,r=R_NEIGH,beta=BETA,eps=EPS)


rng=np.random.default_rng(0)
M=128
cas=[("original (pas de reech.)",None,None),
     ("x1.3 cubique",166,cubic_kernel),
     ("x1.3 lineaire",166,linear_kernel),
     ("x1.3 plus proche voisin",166,kernel_nn),
     ("x1.6 cubique",205,cubic_kernel),
     ("x0.8 cubique (sous-ech.)",102,cubic_kernel)]

res=[]
print(f"{'cas':28s} {'taille':>9s} | {'detecte':>7s} | {'d attendus':>12s} | {'d trouves V':>16s} | {'d trouves H':>16s} | log10 NFA min")
print("-"*125)
for nom,N,ker in cas:
    u=rng.standard_normal((M,M))
    if N is None:
        img=rng.standard_normal((M,M)); att="-"
    else:
        img=resample_2d_sep(u,(N,N),ker); att=f"{M%N},{(-M)%N}"
    t=time.time(); o=pipeline(img); dt=time.time()-t
    lmin=np.log10(min(o["best_v"][1],o["best_h"][1]))
    res.append((nom,img,o,N))
    print(f"{nom:28s} {str(img.shape[0])+'x'+str(img.shape[1]):>9s} | {str(o['detected']):>7s} | {att:>12s} | {str([int(x) for x in o['d_v']]):>16s} | {str([int(x) for x in o['d_h']]):>16s} | {lmin:6.1f}   ({dt:.2f}s)")

# figure : log10 NFA valide (axe V) pour chaque cas
fig,ax=plt.subplots(2,3,figsize=(16,8))
for a,(nom,img,o,N) in zip(ax.ravel(),res):
    with np.errstate(divide="ignore"):
        a.plot(np.log10(o["nfa_v_cv"]),lw=1,label="axe V (valide H/V)")
        a.plot(np.log10(o["nfa_h_cv"]),lw=1,alpha=.7,label="axe H (valide H/V)")
    a.axhline(np.log10(EPS),color="C3",ls="--",lw=1,label="seuil eps=1")
    if N is not None:
        for d in (M%N,(-M)%N):
            a.axvline(d,color="green",ls=":",lw=1)
    a.set_title(f"{nom}  ->  {'DETECTE' if o['detected'] else 'non detecte'}",
                color="C2" if o["detected"]==(N is not None) else "C3")
    a.set_xlabel("distance d"); a.set_ylabel("log10 NFA")
    a.set_ylim(-45,5)
ax[0,0].legend(fontsize=8)
fig.suptitle(f"EXP 1 - bruit blanc {M}x{M}, rank transform 7x7, patch H/10, r={R_NEIGH}, beta={BETA}  (vert pointille = d attendus)",fontsize=12)
fig.tight_layout()
out=os.path.join(os.path.dirname(os.path.abspath(__file__)),"exp_detection_bruit_2d.png")
fig.savefig(out,dpi=120); print("\nfigure ->",out)