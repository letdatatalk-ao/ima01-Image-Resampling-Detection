"""EXP 1 - Detecteur complet sur bruit blanc 2D (rapport Section 4.4).
Bruit uniforme 256x256 reechantillonne en 333x333 (cubique), et un bruit 333x333 non reechantillonne.
Attendu : detection a d = M mod N = 77 et d = -M mod N = 256 sur les deux axes ; rien sur l'original.
On ajoute quelques autres noyaux et facteurs pour comparaison.
Usage : python experiences/exp_detection_bruit_2d.py"""
import sys,os
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.resampling import resample_2d_fast, kernel_nn, linear_kernel, cubic_kernel
from src.detector import detect

rng=np.random.default_rng(0)
M=256
u=rng.uniform(0,255,(M,M))
cas=[("x1.3 cubique (333)",resample_2d_fast(u,(333,333),cubic_kernel),333),
     ("original (333)",rng.uniform(0,255,(333,333)),None),
     ("x1.3 lineaire",resample_2d_fast(u,(333,333),linear_kernel),333),
     ("x1.3 plus proche voisin",resample_2d_fast(u,(333,333),kernel_nn),333),
     ("x0.8 cubique (reduction)",resample_2d_fast(u,(205,205),cubic_kernel),205)]

print(f"{'cas':26s} | detecte | d attendus | d trouves (V)        | log10 NFA min")
for nom,img,N in cas:
    o=detect(img)                          # rank transform 7x7, patch H/10, r=20, beta=0.005, eps=1
    att=f"{M%N},{(-M)%N}" if N else "-"
    print(f"{nom:26s} | {str(o['detected']):7s} | {att:10s} | {str([int(x) for x in o['d_v']]):20s} | {o['log10_nfa']:6.1f}")
