import sys,os
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.resampling import resample_2d_sep,cubic_kernel
from src.spectral import patch_correlations_fast

def colored_noise(shape,seed=0):
    rng=np.random.default_rng(seed)
    w=rng.standard_normal(shape); H,W=shape
    fy=np.fft.fftfreq(H)[:,None]; fx=np.fft.fftfreq(W)[None,:]
    r=np.sqrt(fy**2+fx**2); r[0,0]=1.0
    spec=np.fft.fft2(w)/(1.0+(r*max(H,W)))
    return np.real(np.fft.ifft2(spec))

M1,M2=48,32
N1,N2=128,128
L=16
dmax=min(N1,N2)//2
pic_V_att=M1%N1
pic_H_att=M2%N2

u=colored_noise((M1,M2))
v=resample_2d_sep(u,(N1,N2),cubic_kernel)
S=np.fft.fft2(v)

rho_V=patch_correlations_fast(S,(L,L),dmax,axis=0)
rho_H=patch_correlations_fast(S,(L,L),dmax,axis=1)
mV=rho_V.mean(axis=0)
mH=rho_H.mean(axis=0)
pic_V=int(np.argmax(mV[1:])+1)
pic_H=int(np.argmax(mH[1:])+1)

print(f"patchs={rho_V.shape[0]} dmax={dmax}")
print(f"axe V : pic a d={pic_V:3d} (attendu {pic_V_att}) -> {'OK' if pic_V==pic_V_att else 'KO'}")
print(f"axe H : pic a d={pic_H:3d} (attendu {pic_H_att}) -> {'OK' if pic_H==pic_H_att else 'KO'}")

fig,ax=plt.subplots(1,2,figsize=(11,4))
ax[0].plot(mV); ax[0].axvline(pic_V_att,color="r",ls="--",label=f"M1 mod N1={pic_V_att}")
ax[0].set_title("axe vertical"); ax[0].set_xlabel("distance d"); ax[0].set_ylabel("rho moyen"); ax[0].legend()
ax[1].plot(mH); ax[1].axvline(pic_H_att,color="r",ls="--",label=f"M2 mod N2={pic_H_att}")
ax[1].set_title("axe horizontal"); ax[1].set_xlabel("distance d"); ax[1].legend()
fig.tight_layout()
out=os.path.join(os.path.dirname(__file__),"exp_pic_2d.png")
fig.savefig(out,dpi=110); print("figure ->",out)