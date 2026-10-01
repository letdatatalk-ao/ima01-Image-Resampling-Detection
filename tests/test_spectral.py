import numpy as np

from src.spectral import pcc_complex

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