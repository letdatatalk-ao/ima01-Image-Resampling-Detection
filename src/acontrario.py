import numpy as np
from scipy.stats import binom


def count_votes(rho, r):
    """Nombre de patches dont la corrélation est maximale à la distance d,
    dans le voisinage [d-r, d+r]. Renvoie k[d] indexé par la distance réelle d."""
    n_patches,n_d= rho.shape
    k_d=np.zeros(n_d)                          
    for d in range(r, n_d - r):                  
        for i in range(n_patches):              
            if rho[i, d]== np.max(rho[i, d - r:d + r + 1]):
                k_d[d] +=1
    return k_d


def binomial_tail(k, n, p):
    """P(X >= k) pour X ~ Binomiale(n, p).  sf(k-1) = P(X > k-1) = P(X >= k)."""
    return binom.sf(k - 1, n, p)


def compute_nfa(rho, r):
    """NFA(d) = N_distances * P(K >= k(d)) sous H0, k ~ Bin(n_patches, 1/(2r+1))."""
    n_patches,n_d = rho.shape
    p = 1/(2*r+1)
    distances=range(r, n_d - r)
    N_distances=len(distances)

    k =count_votes(rho, r)                      
    nfa=np.full(n_d, np.inf)                  
    for d in distances:
        nfa[d]=N_distances * binomial_tail(k[d], n_patches, p)
    return nfa


def decide(nfa, eps=1.0):
    """Renvoie les distances détectées : celles dont NFA < eps."""
    return np.where(nfa < eps)[0]