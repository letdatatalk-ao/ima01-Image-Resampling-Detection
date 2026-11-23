"""Modele de reechantillonnage (eq. 2-5 du papier, rapport Sections 2.1, 3.1, 4.1).

    v_j = sum_k u_k phi(j M/N - k),   j = 0..N-1,   u de taille M, v de taille N.

Le signal u est suppose M-periodique (comme dans la theorie) : la distance j M/N - k
est ramenee dans [-M/2, M/2] par wrap. Quatre noyaux phi (Table II du papier).
Deux fonctions :
    resample_1d        boucle directe en 1D (lisible, utilisee pour l'etude 1D)
    resample_2d_fast   2D : le noyau est separable, donc v = W1 u W2^T (deux produits matriciels)
"""
import numpy as np


# ---------------------------------------------------------------- noyaux (Table II)
def kernel_nn(x):
    """Plus proche voisin, support [-1/2, 1/2)."""
    x=np.asarray(x,dtype=float)
    return np.where((x>=-0.5)&(x<0.5),1.0,0.0)


def linear_kernel(x):
    """Lineaire (triangle), support [-1, 1]."""
    x=np.asarray(x,dtype=float)
    return np.where(np.abs(x)<=1.0,1.0-np.abs(x),0.0)


def cubic_kernel(x):
    """Cubique de Keys (a = -1/2), support [-2, 2]."""
    x=np.abs(np.asarray(x,dtype=float))
    return np.where(x<=1,1.5*x**3-2.5*x**2+1,
                    np.where(x<=2,-0.5*x**3+2.5*x**2-4*x+2,0.0))


def lanczos_kernel(x,a=3):
    """Lanczos : sinc(x) sinc(x/a) sur [-a, a] (a = 3)."""
    x=np.asarray(x,dtype=float)
    return np.where(np.abs(x)<=a,np.sinc(x)*np.sinc(x/a),0.0)


# ---------------------------------------------------------------- bord periodique
def wrap(d,M):
    """Ramene une distance d dans [-M/2, M/2] : d - M round(d/M) (signal M-periodique)."""
    d=np.asarray(d,dtype=float)
    return d-M*np.round(d/M)


# ---------------------------------------------------------------- 1D
def resample_1d(u,N,kernel):
    """v_j = sum_k u_k phi(wrap(j M/N - k)), boucle sur les N echantillons de sortie."""
    u=np.asarray(u,dtype=float)
    M=len(u)
    k=np.arange(M)
    v=np.zeros(N,dtype=float)
    for j in range(N):
        x=j*M/N                                   # position de v_j sur la grille de u
        v[j]=np.sum(u*kernel(wrap(x-k,M)))
    return v


# ---------------------------------------------------------------- version matricielle (rapide)
_W_CACHE={}


def resampling_matrix(M,N,kernel):
    """Matrice W (N x M) telle que v = W u : W[j,k] = phi(wrap(j M/N - k)). Mise en cache."""
    key=(M,N,kernel)
    if key not in _W_CACHE:
        x=np.arange(N)[:,None]*M/N-np.arange(M)[None,:]
        _W_CACHE[key]=kernel(wrap(x,M))
    return _W_CACHE[key]


def resample_2d_fast(u,out_shape,kernel):
    """v = W1 u W2^T : resample_1d sur chaque colonne (W1) puis sur chaque ligne (W2), en deux produits matriciels."""
    u=np.asarray(u,dtype=float)
    M1,M2=u.shape; N1,N2=out_shape
    return resampling_matrix(M1,N1,kernel)@u@resampling_matrix(M2,N2,kernel).T
