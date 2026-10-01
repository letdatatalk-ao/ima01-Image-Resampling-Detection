import numpy as np

from src.resampling import kernel_nn, linear_kernel, wrap, resample_1d



def test_resample_x2_nn_formule_spectrale():
    from src.dft import dft
    rng = np.random.default_rng(0)
    M = 16
    u = rng.standard_normal(M)
    N = 2 * M

    v = resample_1d(u, N, kernel_nn)

    n = np.arange(N)
    H = 0.5 * (1 + np.exp(-1j * 2 * np.pi * n / N))   
    v_theorique = H * dft(u)[n % M]                    

    assert np.allclose(np.abs(dft(v)), np.abs(v_theorique), atol=1e-12)