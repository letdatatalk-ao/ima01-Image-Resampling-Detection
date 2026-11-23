"""EXP 2 - Position des pics sur un bruit colore (rapport Section 4.4).
Bruit colore 48x32 (spectre en 1/f, comme une image) reechantillonne en 128x128 (cubique), patch 16x16.
Attendu : correlation moyenne maximale a d = M1 = 48 (axe vertical) et d = M2 = 32 (axe horizontal).
Usage : python experiences/exp_pic_2d.py"""
import sys,os
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.resampling import resample_2d_sep, cubic_kernel
from src.spectral import patch_correlations_fast


def colored_noise(shape,seed=0):
    """Bruit blanc filtre par 1/(1 + |f| max(H,W)) : energie concentree en basses frequences."""
    rng=np.random.default_rng(seed)
    w=rng.standard_normal(shape); H,W=shape
    fy=np.fft.fftfreq(H)[:,None]; fx=np.fft.fftfreq(W)[None,:]
    f=np.sqrt(fy**2+fx**2); f[0,0]=1.0
    return np.real(np.fft.ifft2(np.fft.fft2(w)/(1.0+f*max(H,W))))


M1,M2,N1,N2,L=48,32,128,128,16
dmax=64
S=np.fft.fft2(resample_2d_sep(colored_noise((M1,M2)),(N1,N2),cubic_kernel))
mV=patch_correlations_fast(S,(L,L),dmax,axis=0).mean(axis=0)     # moyenne sur les patches
mH=patch_correlations_fast(S,(L,L),dmax,axis=1).mean(axis=0)
pic_V=int(np.argmax(mV[1:])+1); pic_H=int(np.argmax(mH[1:])+1)  # on saute d = 0 (rho = 1)
print(f"axe vertical   : pic a d = {pic_V} (attendu {M1 % N1})")
print(f"axe horizontal : pic a d = {pic_H} (attendu {M2 % N2})")
