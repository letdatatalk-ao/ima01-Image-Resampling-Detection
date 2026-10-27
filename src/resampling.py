import numpy as np


def kernel_nn(x):
    """nearest neighbor kernel, support [-0.5, 0.5) (Tab. II)"""
    x = np.asarray(x, dtype=float)
    return np.where((x >= -0.5) & (x < 0.5), 1.0, 0.0)


def linear_kernel(x):
    """linear kernel , support [-1, 1] (Tab. II)"""
    x = np.asarray(x, dtype=float)
    return np.where(np.abs(x) <= 1.0, 1.0 - np.abs(x), 0.0)
def cubic_kernel(x):
    """cubic (Keys) kernel for x array, support [-2,2] (Tab. II)"""
    x=np.abs(np.asarray(x, dtype=float))
    out=np.where(x<=1, 1.5*x**3 - 2.5*x**2 + 1,
                 np.where(x<=2, -0.5*x**3 + 2.5*x**2 - 4*x + 2, 0.0))
    return out

def lanczos_kernel(x, a=3):
    """lanczos kernel (a=3) for x array, support [-a,a] (Tab. II)"""
    x=np.asarray(x, dtype=float)
    out=np.sinc(x) * np.sinc(x/a)
    return np.where(np.abs(x)<=a, out, 0.0)

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


def resample_2d(u, out_shape, kernel):
    u=np.asarray(u, dtype=float); M1,M2=u.shape; N1,N2=out_shape
    v=np.zeros((N1,N2), dtype=float); k1=np.arange(M1); k2=np.arange(M2)
    for j1 in range(N1):
        x1=j1*M1/N1                      
        w1=kernel(wrap(x1-k1, M1))
        for j2 in range(N2):
            x2=j2*M2/N2                  
            w2=kernel(wrap(x2-k2, M2))
            v[j1,j2]=np.sum(u * w1[:,None] * w2[None,:])
    return v


def resample_2d_sep(u, out_shape, kernel):
    u=np.asarray(u, dtype=float); M1,M2=u.shape; N1,N2=out_shape
    tmp=np.array([resample_1d(u[i,:], N2, kernel) for i in range(M1)]) 
    out=np.array([resample_1d(tmp[:,j], N1, kernel) for j in range(N2)]).T  
    return out