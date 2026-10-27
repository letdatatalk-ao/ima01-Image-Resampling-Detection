import numpy as np

def pcc_complex(x, y):
    """Pearson correlation coefficient for complex vectors (modulus)."""
    x=np.asarray(x, dtype=complex)
    y=np.asarray(y, dtype=complex)
    xc=x-np.mean(x)
    yc=y-np.mean(y)
    num=np.sum(xc*np.conj(yc))
    den=np.linalg.norm(xc)*np.linalg.norm(yc)
    return np.abs(num)/(den+1e-12)

def patch_correlations_1d(spectrum, L, dmax):
    """1D: rho[patch, d] = pcc(segment, segment shifted by d). Reference."""
    N=len(spectrum)
    n_seg=N//L
    rho=np.zeros((n_seg, dmax+1), dtype=float)
    for i in range(n_seg):
        P=spectrum[i*L+np.arange(L)]
        for d in range(dmax+1):
            idx=(i*L+d+np.arange(L))%N
            rho[i,d]=pcc_complex(P, spectrum[idx])
    return rho

# ----------------------------------------------------------------------
# Version naive 2D (reference lente et lisible) - patch rectangulaire (h,w)
# axis=0 : decalage vertical (lignes),  d = 0..H-1
# axis=1 : decalage horizontal (colonnes), d = 0..W-1
# sortie aplatie (npr*npc, dmax+1), indice i = m*npc + n  -> compatible B4
# ----------------------------------------------------------------------
def patch_correlations_naive(spectrum, patch_shape, dmax, axis=0):
    F=np.asarray(spectrum, dtype=complex)
    H,W=F.shape
    h,w=patch_shape
    npr,npc=H//h, W//w
    rho=np.zeros((npr*npc, dmax+1), dtype=float)
    for m in range(npr):
        for n in range(npc):
            i=m*npc+n
            P=F[m*h:(m+1)*h, n*w:(n+1)*w].ravel()
            for d in range(dmax+1):
                if axis==0:
                    rows=(m*h+d+np.arange(h))%H
                    Q=F[rows, n*w:(n+1)*w].ravel()
                elif axis==1:
                    cols=(n*w+d+np.arange(w))%W
                    Q=F[m*h:(m+1)*h, cols].ravel()
                else:
                    raise ValueError("axis must be 0 or 1")
                rho[i,d]=pcc_complex(P, Q)
    return rho

# ----------------------------------------------------------------------
# Version FFT (moteur rapide) - meme resultat, toutes les distances d'un coup
# ----------------------------------------------------------------------
def _cyclic_window_sum(values, window):
    """Sum of `window` consecutive cyclic values, for every start index."""
    values=np.asarray(values)
    n=values.size
    if not (1<=window<=n):
        raise ValueError("window must be in [1, len(values)]")
    ext=np.concatenate((values, values[:window-1]))
    pref=np.empty(ext.size+1, dtype=np.result_type(values.dtype, np.float64))
    pref[0]=0
    np.cumsum(ext, out=pref[1:])
    return pref[window:]-pref[:-window]

def _correlation_axis(spectrum, patch_shape, axis, eps):
    """FFT cross-correlation of every non-overlapping patch with all cyclic
    shifts on one axis. Returns (npr, npc, length) with length=H (axis0) or W."""
    F=np.asarray(spectrum, dtype=np.complex128)
    H,W=F.shape
    h,w=patch_shape
    npr,npc=H//h, W//w
    length=H if axis==0 else W
    area=float(h*w)
    shifts=np.arange(length)
    rho=np.zeros((npr, npc, length), dtype=np.float64)
    if axis==0:
        sum_win=[]; eng_win=[]
        for n in range(npc):
            c0,c1=n*w,(n+1)*w
            s=F[:,c0:c1].sum(axis=1)
            e=(np.abs(F[:,c0:c1])**2).sum(axis=1)
            sum_win.append(_cyclic_window_sum(s, h))
            eng_win.append(_cyclic_window_sum(e, h))
        for m in range(npr):
            r0=m*h
            rot=np.roll(F, -r0, axis=0)
            pad=np.zeros_like(rot); pad[:h,:]=rot[:h,:]
            A=np.fft.fft(pad, axis=0); B=np.fft.fft(rot, axis=0)
            raw=np.conj(np.fft.ifft(np.conj(A)*B, axis=0))   # raw[d,c]=sum_p P conj(F[r0+p+d])
            for n in range(npc):
                c0,c1=n*w,(n+1)*w
                dot=raw[:,c0:c1].sum(axis=1)
                sw,ew=sum_win[n],eng_win[n]
                q_sum=sw[(r0+shifts)%H]; q_eng=ew[(r0+shifts)%H]
                p_sum=sw[r0]; p_eng=ew[r0]
                cen=dot-p_sum*np.conj(q_sum)/area
                pv=max(float(np.real(p_eng-abs(p_sum)**2/area)), 0.0)
                qv=np.maximum(np.real(q_eng-np.abs(q_sum)**2/area), 0.0)
                rho[m,n,:]=np.clip(np.abs(cen)/(np.sqrt(pv*qv)+eps), 0.0, 1.0)
    elif axis==1:
        sum_win=[]; eng_win=[]
        for m in range(npr):
            r0,r1=m*h,(m+1)*h
            s=F[r0:r1,:].sum(axis=0)
            e=(np.abs(F[r0:r1,:])**2).sum(axis=0)
            sum_win.append(_cyclic_window_sum(s, w))
            eng_win.append(_cyclic_window_sum(e, w))
        for n in range(npc):
            c0=n*w
            rot=np.roll(F, -c0, axis=1)
            pad=np.zeros_like(rot); pad[:,:w]=rot[:,:w]
            A=np.fft.fft(pad, axis=1); B=np.fft.fft(rot, axis=1)
            raw=np.conj(np.fft.ifft(np.conj(A)*B, axis=1))
            for m in range(npr):
                r0,r1=m*h,(m+1)*h
                dot=raw[r0:r1,:].sum(axis=0)
                sw,ew=sum_win[m],eng_win[m]
                q_sum=sw[(c0+shifts)%W]; q_eng=ew[(c0+shifts)%W]
                p_sum=sw[c0]; p_eng=ew[c0]
                cen=dot-p_sum*np.conj(q_sum)/area
                pv=max(float(np.real(p_eng-abs(p_sum)**2/area)), 0.0)
                qv=np.maximum(np.real(q_eng-np.abs(q_sum)**2/area), 0.0)
                rho[m,n,:]=np.clip(np.abs(cen)/(np.sqrt(pv*qv)+eps), 0.0, 1.0)
    else:
        raise ValueError("axis must be 0 or 1")
    return rho

def patch_correlations_fast(spectrum, patch_shape, dmax, axis=0, eps=1e-12):
    """FFT version, aplatie en (npr*npc, dmax+1), meme convention que naive."""
    rho=_correlation_axis(spectrum, patch_shape, axis, eps)
    npr,npc,length=rho.shape
    dmax=min(dmax, length-1)
    return rho.reshape(npr*npc, length)[:, :dmax+1]

def patch_correlations_2d(spectrum, patch_shape=None, patch_ratio=10,
                          dmax=None, eps=1e-12, method="fast"):
    """Correlations 2D sur les deux axes. Renvoie (rho_V, rho_H, patch_shape).
    rho_V : decalage vertical (d=0..dmax_V), rho_H : horizontal (d=0..dmax_H)."""
    F=np.asarray(spectrum, dtype=np.complex128)
    if F.ndim!=2:
        raise ValueError("spectrum must be 2D")
    if not np.all(np.isfinite(F)):
        raise ValueError("spectrum contains NaN or inf")
    H,W=F.shape
    if patch_shape is None:
        h=max(1, H//patch_ratio); w=max(1, W//patch_ratio)
    else:
        h,w=patch_shape
    if not (1<=h<=H and 1<=w<=W):
        raise ValueError("patch must be positive and fit in the spectrum")
    dmax_V=H-1 if dmax is None else min(dmax, H-1)
    dmax_H=W-1 if dmax is None else min(dmax, W-1)
    corr=patch_correlations_fast if method=="fast" else patch_correlations_naive
    if method=="fast":
        rho_V=corr(F, (h,w), dmax_V, axis=0, eps=eps)
        rho_H=corr(F, (h,w), dmax_H, axis=1, eps=eps)
    else:
        rho_V=corr(F, (h,w), dmax_V, axis=0)
        rho_H=corr(F, (h,w), dmax_H, axis=1)
    return rho_V, rho_H, (h,w)


def tv_denoise(image, weight=0.08, iterations=100):
    """TV denoiser for a 2-D grayscale image normalized to [0, 1]."""
    f = np.asarray(image, dtype=np.float64)
    if f.ndim != 2:
        raise ValueError("Expected a 2-D grayscale image")
    if weight < 0 or iterations < 1:
        raise ValueError("weight must be >= 0 and iterations must be >= 1")
    if weight == 0:
        return f.copy()

    u = f.copy()
    u_bar = u.copy()
    px = np.zeros_like(f)  # horizontal dual variable
    py = np.zeros_like(f)  # vertical dual variable
    tau = sigma = 0.25

    for _ in range(iterations):
        # Forward image gradients.
        gx = np.zeros_like(f)
        gy = np.zeros_like(f)
        gx[:, :-1] = u_bar[:, 1:] - u_bar[:, :-1]
        gy[:-1, :] = u_bar[1:, :] - u_bar[:-1, :]

        # Update dual variables and project them onto the TV constraint ball.
        px += sigma * gx
        py += sigma * gy
        norm = np.sqrt(px**2 + py**2)
        scale = np.maximum(1.0, norm / weight)
        px /= scale
        py /= scale

        # Divergence of the dual variables.
        div = np.zeros_like(f)
        div[:, 0] = px[:, 0]
        div[:, 1:] += px[:, 1:] - px[:, :-1]
        div[0, :] += py[0, :]
        div[1:, :] += py[1:, :] - py[:-1, :]

        # Data-fidelity update (Rudin-Osher-Fatemi TV model).
        u_next = (u + tau * div + tau * f) / (1.0 + tau)
        u_bar = 2.0 * u_next - u
        u = u_next

    return np.clip(u, 0.0, 1.0)
