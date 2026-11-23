"""Correlations entre patches du spectre (rapport Sections 2.2, 3.2, 4.2).

rho(x, y) = |<x - mean x, y - mean y>| / (||x - mean x|| ||y - mean y||)   (Pearson complexe, eq. 5)

Pour chaque patch P du spectre et chaque decalage d, on compare P avec la fenetre de meme taille
decalee de d (indices modulo la taille du spectre). Sortie : rho[i, d], i = numero du patch.
    patch_correlations_1d     1D, boucles (reference)
    patch_correlations_naive  2D, boucles (reference lente, sert aux tests et au benchmark)
    patch_correlations_fast   2D, FFT + sommes cumulees (meme resultat, utilise par le detecteur)
"""
import numpy as np


def pcc_complex(x,y):
    """Coefficient de Pearson complexe (module), dans [0, 1]. Invariant a y -> c y, c complexe."""
    x=np.asarray(x,dtype=complex); y=np.asarray(y,dtype=complex)
    xc=x-np.mean(x); yc=y-np.mean(y)
    num=np.sum(xc*np.conj(yc))
    den=np.linalg.norm(xc)*np.linalg.norm(yc)
    return np.abs(num)/(den+1e-12)


# ---------------------------------------------------------------- 1D
def patch_correlations_1d(spectrum,L,dmax):
    """rho[i, d] = pcc(P_i, P_i decale de d), P_i = spectrum[iL : iL+L], d = 0..dmax."""
    N=len(spectrum)
    n_patches=N//L
    rho=np.zeros((n_patches,dmax+1),dtype=float)
    for i in range(n_patches):
        P=spectrum[i*L+np.arange(L)]
        for d in range(dmax+1):
            Q=spectrum[(i*L+d+np.arange(L))%N]        # decalage circulaire
            rho[i,d]=pcc_complex(P,Q)
    return rho


# ---------------------------------------------------------------- 2D, reference
def patch_correlations_naive(spectrum,patch_shape,dmax,axis=0):
    """Patches h x w sans recouvrement, numerotes i = m*npc + n (ligne m, colonne n).
    axis=0 : la fenetre Q est decalee de d lignes ; axis=1 : de d colonnes."""
    F=np.asarray(spectrum,dtype=complex)
    H,W=F.shape; h,w=patch_shape
    npr,npc=H//h,W//w
    rho=np.zeros((npr*npc,dmax+1),dtype=float)
    for m in range(npr):
        for n in range(npc):
            P=F[m*h:(m+1)*h,n*w:(n+1)*w]
            for d in range(dmax+1):
                if axis==0:
                    Q=F[(m*h+d+np.arange(h))%H,n*w:(n+1)*w]
                elif axis==1:
                    Q=F[m*h:(m+1)*h,(n*w+d+np.arange(w))%W]
                else:
                    raise ValueError("axis doit valoir 0 ou 1")
                rho[m*npc+n,d]=pcc_complex(P.ravel(),Q.ravel())
    return rho


# ---------------------------------------------------------------- 2D, rapide
def _cyclic_window_sum(values,window):
    """out[a] = values[a] + ... + values[a+window-1] (indices modulo n), pour tous les a.
    Une seule somme cumulee : out[a] = pref[a+window] - pref[a]."""
    values=np.asarray(values)
    n=values.size
    if not 1<=window<=n:
        raise ValueError("window doit etre dans [1, len(values)]")
    ext=np.concatenate((values,values[:window-1]))       # on recopie le debut a la fin (cyclique)
    pref=np.zeros(ext.size+1,dtype=np.result_type(values.dtype,np.float64))
    np.cumsum(ext,out=pref[1:])
    return pref[window:]-pref[:-window]


def _correlation_axis(spectrum,patch_shape,axis,eps):
    """rho[m, n, d] pour tous les patches (m, n) et tous les decalages d de l'axe choisi.

    On developpe le Pearson (eq. 9 du rapport) :
        rho = |C - S_P conj(S_Q)/A| / sqrt((E_P - |S_P|^2/A)(E_Q - |S_Q|^2/A)),   A = h w
    avec C = sum P conj(Q), S = somme, E = somme des |.|^2 sur la fenetre.
    - S_Q(d), E_Q(d) : sommes sur une fenetre glissante -> _cyclic_window_sum ;
    - C(d) pour tous les d : correlation croisee = produit de polynomes -> FFT."""
    F=np.asarray(spectrum,dtype=np.complex128)
    if axis==1:
        # decalage horizontal = decalage vertical du spectre transpose
        h,w=patch_shape
        return _correlation_axis(F.T,(w,h),0,eps).transpose(1,0,2)
    if axis!=0:
        raise ValueError("axis doit valoir 0 ou 1")
    H,W=F.shape; h,w=patch_shape
    npr,npc=H//h,W//w
    area=float(h*w)
    d=np.arange(H)
    rho=np.zeros((npr,npc,H),dtype=np.float64)

    # 1) somme et energie de toutes les fenetres h x w, pour chaque bloc de colonnes n
    win_sum,win_energy=[],[]
    for n in range(npc):
        block=F[:,n*w:(n+1)*w]
        win_sum.append(_cyclic_window_sum(block.sum(axis=1),h))
        win_energy.append(_cyclic_window_sum((np.abs(block)**2).sum(axis=1),h))

    # 2) pour chaque ligne de patches m : C(d) par FFT, pour toutes les colonnes a la fois
    for m in range(npr):
        r0=m*h
        rot=np.roll(F,-r0,axis=0)                   # le patch commence a la ligne 0
        pad=np.zeros_like(rot); pad[:h]=rot[:h]     # le patch seul, complete par des zeros
        A=np.fft.fft(pad,axis=0); B=np.fft.fft(rot,axis=0)
        C=np.conj(np.fft.ifft(np.conj(A)*B,axis=0)) # C[d, c] = sum_p P[p, c] conj(F[r0+p+d, c])
        for n in range(npc):
            dot=C[:,n*w:(n+1)*w].sum(axis=1)        # somme sur les w colonnes du patch
            s_p,e_p=win_sum[n][r0],win_energy[n][r0]
            s_q,e_q=win_sum[n][(r0+d)%H],win_energy[n][(r0+d)%H]
            num=dot-s_p*np.conj(s_q)/area           # centrage (on retire les moyennes)
            var_p=max(float(np.real(e_p-abs(s_p)**2/area)),0.0)
            var_q=np.maximum(np.real(e_q-np.abs(s_q)**2/area),0.0)
            rho[m,n]=np.clip(np.abs(num)/(np.sqrt(var_p*var_q)+eps),0.0,1.0)
    return rho


def patch_correlations_fast(spectrum,patch_shape,dmax,axis=0,eps=1e-12):
    """Meme sortie que patch_correlations_naive : (npr*npc, dmax+1)."""
    rho=_correlation_axis(spectrum,patch_shape,axis,eps)
    npr,npc,length=rho.shape
    dmax=min(dmax,length-1)
    return rho.reshape(npr*npc,length)[:,:dmax+1]


def patch_correlations_2d(spectrum,patch_shape=None,patch_ratio=10,dmax=None,eps=1e-12,method="fast"):
    """Correlations sur les deux axes. Patch par defaut : (H/patch_ratio, W/patch_ratio).
    Renvoie rho_V (decalages verticaux), rho_H (horizontaux) et la taille de patch (h, w)."""
    F=np.asarray(spectrum,dtype=np.complex128)
    if F.ndim!=2:
        raise ValueError("spectre 2D attendu")
    if not np.all(np.isfinite(F)):
        raise ValueError("le spectre contient NaN ou inf")
    H,W=F.shape
    h,w=patch_shape if patch_shape is not None else (max(1,H//patch_ratio),max(1,W//patch_ratio))
    if not (1<=h<=H and 1<=w<=W):
        raise ValueError("le patch doit etre positif et tenir dans le spectre")
    dmax_V=H-1 if dmax is None else min(dmax,H-1)
    dmax_H=W-1 if dmax is None else min(dmax,W-1)
    if method=="fast":
        rho_V=patch_correlations_fast(F,(h,w),dmax_V,axis=0,eps=eps)
        rho_H=patch_correlations_fast(F,(h,w),dmax_H,axis=1,eps=eps)
    elif method=="naive":
        rho_V=patch_correlations_naive(F,(h,w),dmax_V,axis=0)
        rho_H=patch_correlations_naive(F,(h,w),dmax_H,axis=1)
    else:
        raise ValueError("method doit valoir 'fast' ou 'naive'")
    return rho_V,rho_H,(h,w)


# ---------------------------------------------------------------- limite (rapport Section 7)
def hermitian_half(rho_V,rho_H,shape,patch_shape):
    """Option etudiee dans les limites, NON utilisee pour nos resultats.
    Image reelle -> spectre hermitien F(-u) = conj F(u) : le couple (P, P+d) a un jumeau miroir au meme d,
    avec presque la meme correlation, donc les votes sont comptes deux fois. On garde un patch par paire :
    colonnes 0..npc/2 pour rho_V, lignes 0..npr/2 pour rho_H (experiences/exp_h0_bruit.py)."""
    H,W=shape; h,w=patch_shape
    npr,npc=H//h,W//w
    keep_V=[m*npc+n for m in range(npr) for n in range(npc//2)]
    keep_H=[m*npc+n for m in range(npr//2) for n in range(npc)]
    return rho_V[keep_V],rho_H[keep_H]
