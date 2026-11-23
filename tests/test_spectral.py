from src.spectral import pcc_complex, patch_correlations_1d
from src.resampling import kernel_nn, resample_1d
from src.dft import dft
import numpy as np


def test_autocorrelation_vaut_1():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200) + 1j * rng.standard_normal(200)
    assert np.isclose(pcc_complex(x, x), 1.0, atol=1e-9)

def test_invariance_echelle_et_phase():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(200) + 1j * rng.standard_normal(200)
    for c in [3.0, 2j, -1 + 1j]:
        assert np.isclose(pcc_complex(x, c * x), 1.0, atol=1e-9)

def test_vecteurs_independants_faible_correlation():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(500) + 1j * rng.standard_normal(500)
    y = rng.standard_normal(500) + 1j * rng.standard_normal(500)
    assert pcc_complex(x, y) < 0.2


def test_pic_a_distance_M_mod_N():
    """Bruit blanc rééchantillonné x2.5 : le pic de rho moyen tombe à d = M mod N."""
    rng = np.random.default_rng(0)
    M, N, L = 100, 250, 10
    dmax = N - 1
    u = rng.standard_normal(M)
    v = resample_1d(u, N, kernel_nn)
    rho = patch_correlations_1d(dft(v), L, dmax)
    rho_moyen = rho.mean(axis=0)

    pic = 1 + int(np.argmax(rho_moyen[1:]))
    assert pic == M % N


def test_pas_de_pic_sans_reechantillonnage():
    """Bruit blanc NON rééchantillonné : pas de pic à d = M mod N (reste bas)."""
    rng = np.random.default_rng(0)
    M, N, L = 100, 250, 10
    dmax = N - 1
    w = rng.standard_normal(N)
    rho_ref = patch_correlations_1d(dft(w), L, dmax).mean(axis=0)

    assert rho_ref[M % N] < 0.6

def test_fast_egal_naive_sur_les_deux_axes():
    """FFT + sommes cumulees == formule directe (patches rectangulaires, H != W)."""
    from src.spectral import patch_correlations_naive, patch_correlations_fast
    F = np.fft.fft2(np.random.default_rng(3).standard_normal((60, 45)))
    for axis, dmax in ((0, 59), (1, 44)):
        a = patch_correlations_naive(F, (6, 5), dmax, axis=axis)
        b = patch_correlations_fast(F, (6, 5), dmax, axis=axis)
        assert np.allclose(a, b, atol=1e-10)


def test_somme_fenetre_cyclique():
    from src.spectral import _cyclic_window_sum
    x = np.arange(5.0)
    assert np.allclose(_cyclic_window_sum(x, 2), [1, 3, 5, 7, 4])
