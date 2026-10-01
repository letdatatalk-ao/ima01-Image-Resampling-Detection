import numpy as np

def pcc_complex(x, y):
    """Pearson correlation coefficient for complex numbers"""
    x=np.asarray(x, dtype=complex)
    y=np.asarray(y, dtype=complex)
    xc = x - np.mean(x)
    yc = y - np.mean(y)
    scalar_product = np.sum(xc * np.conj(yc))
    norm_product = np.linalg.norm(xc) * np.linalg.norm(yc)
    return np.abs(scalar_product) / (norm_product + 1e-12)  


def patch_correlations_1d(spectrum, L, dmax):
    """Compute patch correlations for a 1D spectrum"""
    N = len(spectrum)
    n_seg=N//L
    rho = np.zeros((n_seg, dmax+1), dtype=float)

    for d in range(dmax+1):
        for i in range(n_seg):
            idx_P= i*L + np.arange(L)
            idx_Q= (i*L + d+ np.arange(L)) % N

            P=spectrum[idx_P]
            Q=spectrum[idx_Q]
            rho[i,d]=pcc_complex(P,Q)
    return rho

            


