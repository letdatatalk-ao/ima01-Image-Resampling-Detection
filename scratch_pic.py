import numpy as np
from src.resampling import resample_1d, kernel_nn
from src.dft import dft
from src.spectral import patch_correlations_1d

rng = np.random.default_rng(0)

M = 100
N = 250                     
L = 4                      
dmax = N - 1

#  signal rééchantillonné
u = rng.standard_normal(M)              
v = resample_1d(u, N, kernel_nn)        
V = dft(v)
rho = patch_correlations_1d(V, L, dmax)
rho_moyen = rho.mean(axis=0)            

# 2. signal NON rééchantillonné (référence ) 
w = rng.standard_normal(N)              
W = dft(w)
rho_ref = patch_correlations_1d(W, L, dmax).mean(axis=0)

# 3. où est le pic ?
d_attendu = M % N
pic = 1 + np.argmax(rho_moyen[1:])      
print(f"distance attendue (M mod N) : {d_attendu}")
print(f"distance du pic trouvé      : {pic}")
print(f"rho au pic (rééchantillonné): {rho_moyen[pic]:.3f}")
print(f"rho à la même distance (ref): {rho_ref[pic]:.3f}")

# L e l ne pose aucun problem (interes)
#  continuer de faire avec 1d c'est un peux facile tous le projet puis suitcher a 2d. 
#numba c'est une biblio qui permet d'accélérer le code python, mais il faut faire attention à la compatibilité avec les types de données.
# Tv denoiser a faire implimenter mais pas maontenat on essayer de faire de total variation denoiser 