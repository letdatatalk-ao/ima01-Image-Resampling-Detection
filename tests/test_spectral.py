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