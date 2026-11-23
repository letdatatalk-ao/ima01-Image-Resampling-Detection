"""EXP 3 - Images synthetiques de formes geometriques (rapport Section 5.2, Table 6).
40 images 128x128 (degrade + rectangles, disques, triangles, lignes + bruit de capteur sigma = 2),
chacune en version originale et en version reechantillonnee x1.3 (cubique).
Pour chaque residu (none, rt, tv) : taux de faux positifs (originales) et de detection (reechantillonnees).
Usage : python experiences/exp_formes_geometriques.py   (environ 2 min)"""
import sys,os
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.resampling import resample_2d_fast, cubic_kernel
from src.detector import detect

N_IMAGES,M,FACT,NOISE=40,128,1.3,2.0
METHODS=["none","rt","tv"]


def image_formes(M,rng):
    """Fond en degrade + 4 a 8 formes, bruit gaussien, quantification sur 8 bits."""
    yy,xx=np.mgrid[0:M,0:M]/M
    img=60+80*rng.random()*xx+60*rng.random()*yy
    for _ in range(int(rng.integers(4,9))):
        t=rng.integers(0,4); c=rng.uniform(-90,90)              # type de forme et contraste
        cx,cy=rng.uniform(0.1,0.9,2); s=rng.uniform(0.05,0.3)
        if t==0:   mask=(abs(xx-cx)<s)&(abs(yy-cy)<rng.uniform(0.05,0.3))   # rectangle
        elif t==1: mask=(xx-cx)**2+(yy-cy)**2<s**2                          # disque
        elif t==2: mask=(yy>cy)&(yy-cy<s)&(abs(xx-cx)<(yy-cy)*0.8)          # triangle
        else:      mask=abs((xx-cx)*np.cos(c)+(yy-cy)*np.sin(c))<0.01       # ligne
        img=img+c*mask
    img=img+rng.normal(0,NOISE,img.shape)
    return np.clip(np.round(img),0,255)


rng=np.random.default_rng(1)
N=int(round(FACT*M))
fp={m:0 for m in METHODS}; tp={m:0 for m in METHODS}
for k in range(N_IMAGES):
    orig=image_formes(M,rng)
    resized=np.clip(np.round(resample_2d_fast(image_formes(M,rng),(N,N),cubic_kernel)),0,255)
    for m in METHODS:
        fp[m]+=detect(orig,residual=m)["detected"]
        tp[m]+=detect(resized,residual=m)["detected"]
print(f"{N_IMAGES} images {M}x{M}, x{FACT} -> {N}x{N} (cubique)")
print("residu | faux positifs | detection")
for m in METHODS:
    print(f"{m:6s} | {100*fp[m]/N_IMAGES:12.1f} % | {100*tp[m]/N_IMAGES:7.1f} %")
