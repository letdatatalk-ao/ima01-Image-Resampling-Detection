import numpy as np
import matplotlib.pyplot as plt
from src.resampling import resample_1d, kernel_nn
from src.dft import dft
from src.spectral import patch_correlations_1d
from src.acontrario import count_votes, compute_nfa, decide

rng = np.random.default_rng(0)
M, N, L, r = 100, 250, 10, 20
dmax = N - 1
eps = 1.0

# --- cas 1 : signal rééchantillonné ---
u = rng.standard_normal(M)
v = resample_1d(u, N, kernel_nn)
rho = patch_correlations_1d(dft(v), L, dmax)
nfa = compute_nfa(rho, r)

# --- cas 2 : signal NON rééchantillonné (référence) ---
w = rng.standard_normal(N)
rho_ref = patch_correlations_1d(dft(w), L, dmax)
nfa_ref = compute_nfa(rho_ref, r)

print("Ree.  : detectees :", decide(nfa, eps))
print("Ref.  : detectees :", decide(nfa_ref, eps))

# ----------------------------------------------------------------------
# COURBE NFA(d) — l'equivalent 1D de la Fig. 2 du papier
# ----------------------------------------------------------------------
d = np.arange(N)

with np.errstate(divide="ignore"):
    log_nfa     = np.log10(nfa)
    log_nfa_ref = np.log10(nfa_ref)

fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# -- en haut : signal reechantillonne --
ax[0].plot(d, log_nfa, color="C0", lw=1)
ax[0].axhline(np.log10(eps), color="C3", ls="--", lw=1.2, label="seuil log10(eps)=0")
for dd in decide(nfa, eps):
    ax[0].plot(dd, log_nfa[dd], "rv", ms=7)
ax[0].axvline(M % N, color="green", ls=":", lw=1, label=f"M mod N = {M % N}")
ax[0].set_title("NFA - signal REECHANTILLONNE (x2.5, plus proche voisin)")
ax[0].set_ylabel("log10 NFA(d)")
ax[0].legend(fontsize=8)

# -- en bas : reference (non reechantillonne) --
ax[1].plot(d, log_nfa_ref, color="C7", lw=1)
ax[1].axhline(np.log10(eps), color="C3", ls="--", lw=1.2)
for dd in decide(nfa_ref, eps):
    ax[1].plot(dd, log_nfa_ref[dd], "rv", ms=7)
ax[1].set_title("NFA - signal NON reechantillonne (reference)")
ax[1].set_xlabel("distance d")
ax[1].set_ylabel("log10 NFA(d)")

fig.subplots_adjust(hspace=0.3)
fig.savefig("nfa_curve.png", dpi=130)
print("Figure enregistree : nfa_curve.png")