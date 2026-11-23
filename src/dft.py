"""DFT avec la convention du papier (eq. 1) : X_n = (1/N) sum_k x_k exp(-2i pi n k / N).
Rapport : Section 3.1."""
import numpy as np


def dft(x):
    """DFT normalisee par 1/N : X[0] est la moyenne de x."""
    x=np.asarray(x,dtype=complex)
    return np.fft.fft(x)/x.shape[0]


def idft(X):
    """Inverse de dft : x_n = sum_k X_k exp(2i pi n k / N)."""
    X=np.asarray(X,dtype=complex)
    return np.fft.ifft(X)*X.shape[0]
