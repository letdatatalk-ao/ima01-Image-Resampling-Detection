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
