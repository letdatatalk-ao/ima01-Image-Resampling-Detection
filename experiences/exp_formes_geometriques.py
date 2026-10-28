"""EXP 2 - Fiabilite sur images synthetiques de formes geometriques.
Pour chaque image : version ORIGINALE (doit donner 'non detecte') et version
REECHANTILLONNEE x1.3 (doit donner 'detecte'), avec 3 extractions de residu :
  none (pas de residu) | rt (rank transform 7x7) | tv (Chambolle, lambda=1).
Mesure : taux de faux positifs (originales) et taux de detection (reechantillonnees)."""
import sys,os,time
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.resampling import resample_2d_sep,cubic_kernel
from src.residual import extract_residual
from src.spectral import patch_correlations_2d
from src.acontrario import detect_2d

N_IMAGES=20          # nombre d'images generees
M=128                # taille originale
FACT=1.3             # facteur de reechantillonnage
NOISE=2.0            # bruit de capteur (ecart-type, echelle 0-255)
METHODS=["none","rt","tv"]
R_NEIGH,BETA,EPS=20,0.005,1.0


def image_formes(M,rng):
    """Fond en degrade + rectangles, disques, triangles, lignes, puis bruit leger et quantification 8 bits."""
    yy,xx=np.mgrid[0:M,0:M]/M
    img=60+80*rng.random()*xx+60*rng.random()*yy
    for _ in range(int(rng.integers(4,9))):
        t=rng.integers(0,4); c=rng.uniform(-90,90)
        cx,cy=rng.uniform(0.1,0.9,2); s=rng.uniform(0.05,0.3)
        if t==0:   mask=(abs(xx-cx)<s)&(abs(yy-cy)<rng.uniform(0.05,0.3))
        elif t==1: mask=(xx-cx)**2+(yy-cy)**2<s**2
        elif t==2: mask=(yy>cy)&(yy-cy<s)&(abs(xx-cx)<(yy-cy)*0.8)
        else:      mask=abs((xx-cx)*np.cos(c)+(yy-cy)*np.sin(c))<0.01
        img=img+c*mask
    img=img+rng.normal(0,NOISE,img.shape)
    return np.clip(np.round(img),0,255)


def pipeline(img,method):
    R=extract_residual(img,method=method)
    H,W=R.shape
    rv,rh,_=patch_correlations_2d(np.fft.fft2(R-R.mean()),patch_ratio=10)
    o=detect_2d(rv,rh,H,W,r=R_NEIGH,beta=BETA,eps=EPS)
    return o,R


rng=np.random.default_rng(1)
N=int(round(FACT*M)); d_att={M%N,(-M)%N}
stats={m:{"fp":0,"tp":0,"bon_d":0,"lnfa_o":[],"lnfa_r":[]} for m in METHODS}
exemple=None
t0=time.time()
for k in range(N_IMAGES):
    orig=image_formes(M,rng)
    reech=np.clip(np.round(resample_2d_sep(image_formes(M,rng),(N,N),cubic_kernel)),0,255)
    for m in METHODS:
        oo,Ro=pipeline(orig,m)
        orr,Rr=pipeline(reech,m)
        s=stats[m]
        s["fp"]+=oo["detected"]; s["tp"]+=orr["detected"]
        s["bon_d"]+=orr["detected"] and bool(set(int(x) for x in orr["d_v"])&d_att)
        s["lnfa_o"].append(np.log10(min(oo["best_v"][1],oo["best_h"][1])))
        s["lnfa_r"].append(np.log10(min(orr["best_v"][1],orr["best_h"][1])))
        if k==0:
            exemple=exemple or {}
            exemple[m]=(Ro,Rr,oo,orr)
    if k==0: ex_orig,ex_reech=orig,reech
    print(f"image {k+1:2d}/{N_IMAGES} traitee ({time.time()-t0:.1f}s)")

print(f"\n=== {N_IMAGES} images {M}x{M}, reechantillonnees x{FACT} -> {N}x{N} (cubique), d attendus {sorted(d_att)} ===")
print(f"{'residu':6s} | {'faux positifs (orig.)':>22s} | {'detection (reech.)':>19s} | {'bon d trouve':>12s} | {'med log10NFA orig':>17s} | {'med log10NFA reech':>18s}")
print("-"*110)
for m in METHODS:
    s=stats[m]
    print(f"{m:6s} | {s['fp']:3d}/{N_IMAGES} = {100*s['fp']/N_IMAGES:5.1f} %    | {s['tp']:3d}/{N_IMAGES} = {100*s['tp']/N_IMAGES:5.1f} % | {s['bon_d']:3d}/{N_IMAGES}    "
          f"| {np.median(s['lnfa_o']):17.1f} | {np.median(s['lnfa_r']):18.1f}")

# ---------------- figure ----------------
fig=plt.figure(figsize=(18,11))
g=fig.add_gridspec(3,5)
a=fig.add_subplot(g[0,0]); a.imshow(ex_orig,cmap="gray"); a.set_title("exemple original"); a.axis("off")
a=fig.add_subplot(g[1,0]); a.imshow(ex_reech,cmap="gray"); a.set_title(f"exemple reech. x{FACT}"); a.axis("off")
for j,m in enumerate(METHODS):
    Ro,Rr,oo,orr=exemple[m]
    a=fig.add_subplot(g[0,1+j]); a.imshow(Rr,cmap="gray"); a.set_title(f"residu '{m}' (reech.)"); a.axis("off")
    a=fig.add_subplot(g[1,1+j])
    with np.errstate(divide="ignore"):
        a.plot(np.log10(oo["nfa_v_cv"]),lw=1,color="C7",label="original")
        a.plot(np.log10(orr["nfa_v_cv"]),lw=1,color="C0",label="reechantillonne")
    a.axhline(0,color="C3",ls="--",lw=1)
    for d in d_att: a.axvline(d,color="green",ls=":",lw=1)
    a.set_title(f"log10 NFA axe V - residu '{m}'"); a.set_xlabel("d"); a.legend(fontsize=7)
# barres FP / detection
a=fig.add_subplot(g[2,:3]); x=np.arange(len(METHODS)); w=0.35
a.bar(x-w/2,[100*stats[m]["fp"]/N_IMAGES for m in METHODS],w,color="C3",label="faux positifs (originales)")
a.bar(x+w/2,[100*stats[m]["tp"]/N_IMAGES for m in METHODS],w,color="C2",label="detections (reechantillonnees)")
a.set_xticks(x); a.set_xticklabels(METHODS); a.set_ylabel("%"); a.set_ylim(0,105); a.legend(); a.set_title("Fiabilite par methode de residu")
# distribution des log NFA
a=fig.add_subplot(g[2,3:])
for j,m in enumerate(METHODS):
    a.scatter(np.full(N_IMAGES,j)-0.1,stats[m]["lnfa_o"],color="C7",s=12)
    a.scatter(np.full(N_IMAGES,j)+0.1,stats[m]["lnfa_r"],color="C0",s=12)
a.axhline(0,color="C3",ls="--",lw=1); a.set_xticks(range(len(METHODS))); a.set_xticklabels(METHODS)
a.set_ylabel("log10 NFA min"); a.set_title("gris = originales, bleu = reechantillonnees")
fig.suptitle(f"EXP 2 - {N_IMAGES} images de formes geometriques, x{FACT} cubique, eps={EPS}",fontsize=13)
fig.tight_layout()
out=os.path.join(os.path.dirname(os.path.abspath(__file__)),"exp_formes_geometriques.png")
fig.savefig(out,dpi=110); print("\nfigure ->",out)