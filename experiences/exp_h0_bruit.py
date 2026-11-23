"""Limite du controle des fausses alarmes (rapport Section 7) : bruit blanc 600x600, aucun reechantillonnage.
Theorie (Sec. III-B) : si les votes des patches sont independants, k(d) ~ B(#P, 1/(2r+1)) et
E[#{d : NFA(d) < eps}] <= eps par axe.  Spectre hermitien -> chaque couple (P, P+d) a un jumeau
conjugue : avec tous les patches (papier) les votes sont redondants. src.spectral.hermitian_half
ne garde qu'un patch par paire.  Usage : python experiences/exp_h0_bruit.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.residual import extract_residual
from src.spectral import patch_correlations_2d, hermitian_half
from src.acontrario import compute_nfa, count_votes, detect_2d

N, R, EPS, N_IMG = 600, 20, 1.0, 50
p = 1/(2*R+1)
rng = np.random.default_rng(0)
st = {m: {"fa": [], "disp": [], "img": [], "cv": []} for m in ("papier", "corrige")}
for t in range(N_IMG):
    img = rng.uniform(0, 255, (N, N))
    res = extract_residual(img, method="rt", s=3)
    rv, rh, hw = patch_correlations_2d(np.fft.fft2(res-res.mean()), patch_ratio=10)
    for m, (a, b) in (("papier", (rv, rh)), ("corrige", hermitian_half(rv, rh, (N, N), hw))):
        n_img = 0
        for rho in (a, b):
            nfa = compute_nfa(rho, R); st[m]["fa"].append(np.sum(nfa < EPS)); n_img += np.sum(nfa < EPS)
            k = count_votes(rho, R)[R+1:N-R]; st[m]["disp"].append(k.var()/(k.mean()*(1-p)))
        st[m]["img"].append(n_img > 0)
        st[m]["cv"].append(detect_2d(a, b, N, N, r=R, eps=EPS)["detected"])
print(f"{N_IMG} images de bruit blanc {N}x{N}, rank transform 7x7, r={R}, eps={EPS}")
for m, s in st.items():
    print(f"{m:8s} #P={'100' if m == 'papier' else ' 50'} | E[#FA] par axe = {np.mean(s['fa']):.2f} (theorie <= {EPS:g})"
          f" | surdispersion des votes = {np.mean(s['disp']):.2f} (binomiale = 1)"
          f" | images avec >=1 FA sans val. croisee = {100*np.mean(s['img']):.0f} %"
          f" | avec val. croisee = {100*np.mean(s['cv']):.0f} %")
