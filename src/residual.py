import numpy as np
 
 
def _check(image):
    """Image 2D (niveaux de gris), float64, valeurs finies."""
    f=np.asarray(image,dtype=np.float64)
    if f.ndim!=2:
        raise ValueError("image 2D (niveaux de gris) attendue")
    if f.size==0 or not np.all(np.isfinite(f)):
        raise ValueError("image vide ou non finie")
    return f
 
 
# ----------------------------------------------------------------------
# Rank transform (eq. 15)
# ----------------------------------------------------------------------
def rank_transform_2d(image,s=3,mode="reflect"):
    """R(x)=#{x' in N(x) : I(x')<I(x)}, fenetre (2s+1)x(2s+1).
    s=3 -> 7x7 (PNG), s=1 -> 3x3 (JPEG). Bords par miroir : pas de cadre a 0."""
    img=_check(image)
    h,w=img.shape
    pad=np.pad(img,s,mode=mode)
    out=np.zeros((h,w),dtype=float)
    for di in range(-s,s+1):
        for dj in range(-s,s+1):
            out+=pad[s+di:s+di+h,s+dj:s+dj+w]<img
    return out
 
 
# ----------------------------------------------------------------------
# TV denoiser - projection de Chambolle (Duran, Coll, Sbert, IPOL 2013 [47])
# min_u TV(u) + lambda/2 ||u-f||^2
# ----------------------------------------------------------------------
def forward_gradient(u):
    """Differences avant (vertical gx, horizontal gy), nulles au bord bas/droit."""
    gx=np.zeros_like(u); gy=np.zeros_like(u)
    if u.shape[0]>1:
        gx[:-1,:]=u[1:,:]-u[:-1,:]
    if u.shape[1]>1:
        gy[:,:-1]=u[:,1:]-u[:,:-1]
    return gx,gy
 
 
def divergence(px,py):
    """Divergence discrete adjointe : <p,grad u> = -<div p,u>."""
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
    """Algorithme de projection de Chambolle. Image a l'echelle [0,255] (lam=1, Table III).
    p <- (p + dt*grad(div p - lam f)) / (1 + dt*|grad(div p - lam f)|),  u = f - div(p)/lam."""
    f=_check(image)
    if lam<=0 or not np.isfinite(lam):
        raise ValueError("lam doit etre fini et > 0")
    if not (0<dt<0.25):
        raise ValueError("dt doit verifier 0<dt<1/4")
    px=np.zeros_like(f); py=np.zeros_like(f)
    converged=False; change=np.inf; it=0
    for it in range(1,max_iter+1):
        q=divergence(px,py)-lam*f
        qx,qy=forward_gradient(q)
        den=1.0+dt*np.sqrt(qx*qx+qy*qy)
        px_new=(px+dt*qx)/den
        py_new=(py+dt*qy)/den
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
    """Residu TV : R = I - TV(I)  (contient bruit + oscillations du reechantillonnage)."""
    f=_check(image)
    return f-tv_denoise_chambolle(f,lam=lam,**kw)
 
 
# ----------------------------------------------------------------------
# Point d'entree unique
# ----------------------------------------------------------------------
def extract_residual(image,method="rt",**kw):
    """method='rt' (PNG, defaut), 'tv' (JPEG), 'none'."""
    if method=="rt":
        return rank_transform_2d(image,**kw)
    if method=="tv":
        return tv_residual(image,**kw)
    if method=="none":
        return _check(image).copy()
    raise ValueError("method doit etre 'rt', 'tv' ou 'none'")
