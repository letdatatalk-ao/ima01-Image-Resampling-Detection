"""Tests de la theorie (Theoreme 1) et du controle des fausses alarmes."""
import numpy as np
from src.resampling import kernel_nn, linear_kernel, resample_1d, resample_2d_fast, resample_2d_sep
from src.acontrario import compute_nfa


def test_fast_egal_separable():
    """resample_2d_fast (produit matriciel) == resample_2d_sep (boucles), noyau lineaire."""
    u = np.random.default_rng(0).standard_normal((12, 15))
    assert np.allclose(resample_2d_fast(u, (17, 11), linear_kernel), resample_2d_sep(u, (17, 11), linear_kernel), atol=1e-12)


def test_theoreme1_copie_spectrale():
    """Agrandissement M -> N (plus proche voisin) : |v^(n)| et |v^(n+M)| proportionnels a |u^(n mod M)|
    -> le motif spectral se repete a la distance d = M mod N (Theoreme 1)."""
    M, N = 100, 250
    u = np.random.default_rng(1).standard_normal(M)
    V = np.fft.fft(resample_1d(u, N, kernel_nn)); U = np.fft.fft(u)
    n = np.arange(5, 45)
    r1, r2 = V[n] / U[n % M], V[n+M] / U[(n+M) % M]   # = M * c_n(phi) : ne depend que du noyau
    assert np.corrcoef(np.abs(V[n]), np.abs(V[n+M]) * np.abs(r1) / np.abs(r2))[0, 1] > 0.999


def test_nfa_h0_votes_independants():
    """Si les patches sont independants (rho tire au hasard), E[#FA] <= eps (Sec. III-B)."""
    rng = np.random.default_rng(2)
    fa = [np.sum(compute_nfa(rng.random((100, 600)), 20) < 1.0) for _ in range(30)]
    assert np.mean(fa) <= 1.0


def test_hermitian_half_supprime_la_redondance():
    """Bruit blanc : un patch par paire conjuguee -> votes binomiaux (surdispersion ~1)."""
    from src.spectral import patch_correlations_2d, hermitian_half
    from src.acontrario import count_votes
    rng = np.random.default_rng(3); d = []
    for t in range(4):
        x = rng.standard_normal((300, 300))
        rv, rh, hw = patch_correlations_2d(np.fft.fft2(x-x.mean()), patch_ratio=10)
        for rho in hermitian_half(rv, rh, x.shape, hw):
            k = count_votes(rho, 20)[21:279]; d.append(k.var()/(k.mean()*(1-1/41)))
    assert np.mean(d) < 1.15
