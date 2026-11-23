"""Les 5 verifications essentielles du detecteur.  Lancer depuis code_ima :  python -m pytest -q"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.dft import dft
from src.resampling import resample_1d, resample_2d_fast, kernel_nn, cubic_kernel
from src.spectral import pcc_complex, patch_correlations_naive, patch_correlations_fast
from src.acontrario import compute_nfa, binomial_tail
from src.detector import detect


def test_reechantillonnage_2d_egal_1d_sur_lignes_et_colonnes():
    """resample_2d_fast (v = W1 u W2^T) == resample_1d applique aux colonnes puis aux lignes."""
    u = np.random.default_rng(0).standard_normal((12, 9))
    cols = np.array([resample_1d(u[:, j], 17, cubic_kernel) for j in range(9)]).T        # (17, 9)
    ref = np.array([resample_1d(cols[i, :], 11, cubic_kernel) for i in range(17)])       # (17, 11)
    assert np.allclose(resample_2d_fast(u, (17, 11), cubic_kernel), ref, atol=1e-12)


def test_le_spectre_contient_une_copie_a_d_egal_M():
    """Theoreme de repliement : apres reechantillonnage M -> N, le patch a d = M est (presque) proportionnel."""
    M, N, L = 100, 250, 10
    V = dft(resample_1d(np.random.default_rng(0).standard_normal(M), N, kernel_nn))
    assert pcc_complex(V[40:50], V[40 + M:50 + M]) > 0.9          # la copie
    assert pcc_complex(V[40:50], V[77:87]) < 0.7                  # un segment quelconque


def test_version_fft_egale_version_naive():
    """Notre moteur rapide (FFT + sommes cumulees) donne exactement la formule directe, sur les deux axes."""
    F = np.fft.fft2(np.random.default_rng(3).standard_normal((60, 45)))
    for axis, dmax in ((0, 59), (1, 44)):
        naive = patch_correlations_naive(F, (6, 5), dmax, axis=axis)
        fast = patch_correlations_fast(F, (6, 5), dmax, axis=axis)
        assert np.max(np.abs(naive - fast)) < 1e-12


def test_nfa_binomiale():
    """P[B(n,p) >= 0] = 1 ; et un rho ou tous les patches votent pour d = 30 donne un NFA minuscule a d = 30."""
    assert binomial_tail(0, 25, 1 / 41) == 1.0
    rho = np.random.default_rng(1).uniform(0, 0.5, (25, 100))
    rho[:, 30] = 0.99                                             # tous les patches votent pour d = 30
    nfa = compute_nfa(rho, r=10)
    assert nfa[30] < 1e-30 and np.argmin(nfa) == 30


def test_detecteur_complet_sur_bruit_blanc():
    """256 -> 333 (cubique) : detecte a 77 ou 256 ; bruit non reechantillonne : non detecte."""
    rng = np.random.default_rng(0)
    v = resample_2d_fast(rng.uniform(0, 255, (256, 256)), (333, 333), cubic_kernel)
    o = detect(v)
    assert o["detected"] and ({77, 256} & set(int(d) for d in o["d_v"]))
    assert not detect(rng.uniform(0, 255, (333, 333)))["detected"]
