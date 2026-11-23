"""Extraction du residu : rapprocher une image naturelle d'un bruit blanc (rapport Section 5.2).
    rt   : rank transform 7x7 (defaut, images PNG)
    tv   : I - u*, u* debruitee par variation totale (projection de Chambolle), images JPEG
    none : l'image elle-meme (pour l'ablation)
Les images sont en float64 sur l'echelle [0, 255] (lambda = 1 est donne pour cette echelle)."""
import numpy as np


def _check(image):
    """Image 2D, float64, non vide, valeurs finies."""
    f=np.asarray(image,dtype=np.float64)
    if f.ndim!=2:
        raise ValueError("image 2D (niveaux de gris) attendue")
    if f.size==0 or not np.all(np.isfinite(f)):
        raise ValueError("image vide ou non finie")
    return f


# ---------------------------------------------------------------- rank transform
def rank_transform_2d(image,s=3,mode="reflect"):
    """R(x) = #{x' dans x + [-s, s]^2 : I(x') < I(x)}. s=3 -> fenetre 7x7. Bords par miroir."""
    img=_check(image)
    h,w=img.shape
    pad=np.pad(img,s,mode=mode)
    out=np.zeros((h,w),dtype=float)
    for di in range(-s,s+1):                 # on parcourt les (2s+1)^2 voisins
        for dj in range(-s,s+1):
            out+=pad[s+di:s+di+h,s+dj:s+dj+w]<img
    return out


# ---------------------------------------------------------------- variation totale (Chambolle)
def forward_gradient(u):
    """Differences avant : gx vertical, gy horizontal, nulles sur la derniere ligne / colonne."""
    gx=np.zeros_like(u); gy=np.zeros_like(u)
    if u.shape[0]>1:
        gx[:-1,:]=u[1:,:]-u[:-1,:]
    if u.shape[1]>1:
        gy[:,:-1]=u[:,1:]-u[:,:-1]
    return gx,gy


def divergence(px,py):
    """Divergence discrete, adjointe de -forward_gradient : <p, grad u> = -<div p, u>."""
    h,w=px.shape
    div=np.zeros_like(px)
    if h>1:
        div[0,:]+=px[0,:]
        if h>2:
            div[1:-1,:]+=px[1:-1,:]-px[:-2,:]
        div[-1,:]-=px[-2,:]
    if w>1:
        div[:,0]+=py[:,0]
        if w>2:
            div[:,1:-1]+=py[:,1:-1]-py[:,:-2]
        div[:,-1]-=py[:,-2]
    return div


def tv_denoise_chambolle(image,lam=1.0,dt=0.248,tol=1e-2,max_iter=10000,return_info=False):
    """u* = argmin TV(u) + lam/2 ||u - f||^2 par la projection de Chambolle (variable duale p) :
        g = grad(div p - lam f),   p <- (p + dt g) / (1 + dt |g|),   u = f - div(p) / lam.
    Arret quand p bouge de moins de tol (max sur les pixels)."""
    f=_check(image)
    if lam<=0 or not np.isfinite(lam):
        raise ValueError("lam doit etre fini et > 0")
    if not 0<dt<0.25:
        raise ValueError("dt doit verifier 0 < dt < 1/4")
    px=np.zeros_like(f); py=np.zeros_like(f)
    converged=False; change=np.inf; it=0
    for it in range(1,max_iter+1):
        gx,gy=forward_gradient(divergence(px,py)-lam*f)
        den=1.0+dt*np.sqrt(gx*gx+gy*gy)
        px_new=(px+dt*gx)/den
        py_new=(py+dt*gy)/den
        change=float(np.max(np.sqrt((px_new-px)**2+(py_new-py)**2)))
        px,py=px_new,py_new
        if change<=tol:
            converged=True
            break
    u=f-divergence(px,py)/lam
    if return_info:
        return u,{"iterations":it,"converged":converged,"final_max_dual_change":change}
    return u


def tv_residual(image,lam=1.0,**kw):
    """R = f - u* : garde le bruit et les oscillations du reechantillonnage."""
    f=_check(image)
    return f-tv_denoise_chambolle(f,lam=lam,**kw)


# ---------------------------------------------------------------- point d'entree
def extract_residual(image,method="rt",**kw):
    """method = 'rt' (PNG, defaut), 'tv' (JPEG) ou 'none'."""
    if method=="rt":
        return rank_transform_2d(image,**kw)
    if method=="tv":
        return tv_residual(image,**kw)
    if method=="none":
        return _check(image).copy()
    raise ValueError("method doit valoir 'rt', 'tv' ou 'none'")
