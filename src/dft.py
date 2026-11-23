"DFT convention du papier (éq. 1)"

import numpy as np
def dft(x):
    """X_n = (1/N) * sum_k x_k * exp(-i 2*pi n k / N)"""

    x=np.asarray(x, dtype=complex)
    return np.fft.fft(x)/x.shape[0]


def idft(X):
    """x_n = sum_k X_k * exp(i 2*pi n k / N)"""
    X=np.asarray(X, dtype=complex)
    return np.fft.ifft(X)*X.shape[0]