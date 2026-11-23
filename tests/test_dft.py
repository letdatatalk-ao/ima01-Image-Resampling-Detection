import numpy as np
from src.dft import dft, idft


def test_dft_inverse_identity():
    """Test that the DFT and IDFT are inverses of each other."""
    x = np.random.rand(10) + 1j * np.random.rand(10)
    X = dft(x)
    x_reconstructed = idft(X)
    assert np.allclose(x, x_reconstructed, atol=1e-12), "DFT and IDFT are not inverses of each other."



def test_dc_is_mean():
    """Test that the DC component of the DFT is equal to the mean of the input signal."""
    x = np.random.rand(10) + 1j * np.random.rand(10)
    X = dft(x)
    dc_component = X[0]
    mean_value = np.mean(x)
    assert np.allclose(dc_component, mean_value, atol=1e-12), "DC component is not equal to the mean of the input signal."
    