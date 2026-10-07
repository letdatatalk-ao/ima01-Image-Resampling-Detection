import numpy as np
from src.resampling import resample_1d, kernel_nn
from src.dft import dft
from src.spectral import patch_correlations_1d
from src.acontrario import compute_nfa, decide


def test_vrai_positif_detecte_M_mod_N():
    """Bruit blanc rééchantillonné"""
    rng = np.random.default_rng(0)
    M,N,L,r = 100, 250, 10, 20
    dmax=N - 1
    u=rng.standard_normal(M)
    v=resample_1d(u,N,kernel_nn)
    rho=patch_correlations_1d(dft(v),L,dmax)
    nfa=compute_nfa(rho,r)
    detections = decide(nfa, eps=1.0)
    assert (M % N) in detections


def test_vrai_negatif_avec_seuil_resserre():
    """Bruit blanc NON rééchantillonné"""
    rng = np.random.default_rng(0)
    M,N,L,r = 100,250,10,20
    dmax= N-1
    w= rng.standard_normal(N)
    rho_ref = patch_correlations_1d(dft(w), L, dmax)
    nfa_ref = compute_nfa(rho_ref, r)
    detections = decide(nfa_ref, eps=1e-3)
    assert len(detections) == 0