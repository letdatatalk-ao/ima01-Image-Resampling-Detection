import numpy as np


def kernel_nn(x):
    """nearest neighbor kernel, support [-0.5, 0.5) (Tab. II)"""
    x = np.asarray(x, dtype=float)
    return np.where((x >= -0.5) & (x < 0.5), 1.0, 0.0)


def linear_kernel(x):
    """linear kernel , support [-1, 1] (Tab. II)"""
    x = np.asarray(x, dtype=float)
    return np.where(np.abs(x) <= 1.0, 1.0 - np.abs(x), 0.0)

def wrap(d, M):
    d=np.asarray(d, dtype=float)
    return (d-M*np.round(d/M))

def resample_1d(u, N, kernel):
    u = np.asarray(u, dtype=float)
    M = len(u)
    v = np.zeros(N, dtype=float)
    k = np.arange(M)                       
    for j in range(N):
        x = j * M / N                      
        v[j] = np.sum(u * kernel(wrap(x - k, M)))   
    return v
