"""Detecteur complet (Fig. 7 du papier) :
image -> residu (rank transform / TV) -> DFT 2D -> correlations de patches (V et H)
      -> votes + NFA (Eq. 17-21) -> validation croisee H/V (Eq. 23) -> decision NFA < eps."""
import numpy as np
from src.image_io import load_gray
from src.residual import extract_residual
from src.spectral import patch_correlations_2d
from src.acontrario import detect_2d

DEFAULTS=dict(residual="rt",s=3,patch_ratio=10,r=20,beta=0.005,eps=1.0,cross_val=True)


def detect(image,residual="rt",s=3,patch_ratio=10,r=20,beta=0.005,eps=1.0,cross_val=True,jpeg=False):
    """image : chemin de fichier ou tableau 2D (niveaux de gris, echelle [0,255]).
    residual : 'rt' (rank transform, fenetre (2s+1)^2 ; s=3 pour PNG, s=1 pour JPEG), 'tv' ou 'none'.
    Renvoie un dict : detected, d_v, d_h (distances validees), best_v/best_h (d, NFA),
    log10_nfa (min des deux axes), nfa_v_cv/nfa_h_cv (courbes), size."""
    img=load_gray(image) if isinstance(image,str) else np.asarray(image,dtype=np.float64)
    if img.ndim!=2:
        raise ValueError("image 2D attendue (utilisez load_gray pour une image couleur)")
    kw={"s":s} if residual=="rt" else {}
    R=extract_residual(img,method=residual,**kw)
    H,W=R.shape
    rho_v,rho_h,_=patch_correlations_2d(np.fft.fft2(R-R.mean()),patch_ratio=patch_ratio)
    out=detect_2d(rho_v,rho_h,H,W,r=r,beta=beta,eps=eps,cross_val=cross_val,jpeg=jpeg)
    best=min(out["best_v"][1],out["best_h"][1])
    with np.errstate(divide="ignore"):
        out["log10_nfa"]=float(np.log10(best)) if best>0 else -np.inf
    with np.errstate(divide="ignore"):
        out["log10_nfa_nocv"]=float(np.log10(out["min_nfa_nocv"])) if out["min_nfa_nocv"]>0 else -np.inf
    out["size"]=(H,W)
    return out


def matches(found,expected,tol=1.0):
    """True si une distance validee est a moins de tol d'une distance attendue."""
    return any(abs(f-e)<=tol for f in found for e in expected)


def harmonic_distances(q,N,kmax=4):
    """Distances +-k*N/q mod N, k=1..kmax (le plus proche voisin produit des harmoniques)."""
    m=N/q
    return sorted({float((s*k*m)%N) for k in range(1,kmax+1) for s in (1,-1)})