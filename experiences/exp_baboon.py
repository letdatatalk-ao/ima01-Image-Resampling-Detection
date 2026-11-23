"""Validation sur les images de reference des auteurs (dossier img/ de github.com/li-yanhao/ird).
baboon.png (512) original, baboon_666.png (512 -> 666), pashmina.jpg (800) original, pashmina_720.jpg (800 -> 720, JPEG).
Attendu : rien sur les originaux, pics a d = +-M mod N sur les reechantillonnees."""
import sys,os
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.image_io import load_gray
from src.detector import detect

IMG=sys.argv[1] if len(sys.argv)>1 else "data/ird_img"
CASES=[("baboon.png",None),("baboon_666.png",512),("pashmina.jpg",None),("pashmina_720.jpg",800)]

fig,ax=plt.subplots(1,4,figsize=(18,3.6))
for a,(f,M) in zip(ax,CASES):
    g=load_gray(os.path.join(IMG,f)); jpg=f.endswith(".jpg"); N=g.shape[0]
    o=detect(g,residual="tv" if jpg else "rt",s=3,jpeg=jpg)
    att=sorted({M%N,(-M)%N}) if M else []
    print(f"{f:18s} {g.shape} detecte={o['detected']!s:5s} d_V={list(map(int,o['d_v']))} d_H={list(map(int,o['d_h']))} attendu={att} log10NFA={o['log10_nfa']:.1f}")
    with np.errstate(divide="ignore"):
        a.plot(np.log10(o["nfa_v_cv"]),lw=1,label="vertical"); a.plot(np.log10(o["nfa_h_cv"]),lw=1,label="horizontal")
    a.axhline(0,color="r",ls="--",lw=1)
    for d in att: a.axvline(d,color="g",ls=":")
    a.set_title(f"{f} : {'detecte' if o['detected'] else 'non detecte'}"); a.set_xlabel("d")
ax[0].set_ylabel("log10 NFA (valide H/V)"); ax[0].legend(fontsize=8)
plt.tight_layout(); out=os.path.join(os.path.dirname(os.path.abspath(__file__)),"exp_baboon.png"); plt.savefig(out,dpi=120); print("figure ->",out)