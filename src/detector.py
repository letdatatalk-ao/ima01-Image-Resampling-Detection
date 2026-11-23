"""Detecteur complet (rapport Section 4, Figure 1) :
image -> residu -> FFT2(R - moyenne) -> correlations des patches (axes V et H)
      -> votes + NFA sur chaque axe -> validation croisee H/V -> detecte si min NFA < eps."""
import numpy as np
from src.image_io import load_gray
from src.residual import extract_residual
from src.spectral import patch_correlations_2d, hermitian_half
from src.acontrario import detect_2d


def detect(image,residual="rt",s=3,patch_ratio=10,r=20,beta=0.005,eps=1.0,cross_val=True,
           jpeg=False,jpeg_block=8,hermitien=False):
    """image : chemin de fichier ou tableau 2D en niveaux de gris sur [0, 255].
    residual    'rt' (rank transform (2s+1)x(2s+1)), 'tv' ou 'none'
    patch_ratio patch = (H/patch_ratio, W/patch_ratio)
    r, eps      rayon des votes et seuil du NFA ;  beta : tolerance de la validation croisee
    jpeg        True -> les distances kN/jpeg_block ne votent pas
    hermitien   False (defaut) : tous les patches votent, comme le papier (nos resultats).
                True : un patch par paire miroir (option des limites, rapport Section 7).
    Renvoie le dict de detect_2d, plus log10_nfa (min des deux axes apres validation croisee),
    log10_nfa_nocv (sans validation croisee) et size = (H, W)."""
    img=load_gray(image) if isinstance(image,str) else np.asarray(image,dtype=np.float64)
    if img.ndim!=2:
        raise ValueError("image 2D attendue (utilisez load_gray pour une image couleur)")
    kw={"s":s} if residual=="rt" else {}
    R=extract_residual(img,method=residual,**kw)
    H,W=R.shape
    rho_v,rho_h,hw=patch_correlations_2d(np.fft.fft2(R-R.mean()),patch_ratio=patch_ratio)
    if hermitien:
        rho_v,rho_h=hermitian_half(rho_v,rho_h,(H,W),hw)
    out=detect_2d(rho_v,rho_h,H,W,r=r,beta=beta,eps=eps,cross_val=cross_val,jpeg=jpeg,jpeg_block=jpeg_block)
    with np.errstate(divide="ignore"):                     # log10(0) = -inf si NFA = 0
        out["log10_nfa"]=float(np.log10(min(out["best_v"][1],out["best_h"][1])))
        out["log10_nfa_nocv"]=float(np.log10(out["min_nfa_nocv"]))
    out["size"]=(H,W)
    return out


def matches(found,expected,tol=1.0):
    """True si une distance trouvee est a moins de tol d'une distance attendue."""
    return any(abs(f-e)<=tol for f in found for e in expected)


def harmonic_distances(q,N,kmax=4):
    """Distances +-k N/q mod N, k = 1..kmax (le plus proche voisin donne aussi les harmoniques)."""
    m=N/q
    return sorted({float((s*k*m)%N) for k in range(1,kmax+1) for s in (1,-1)})
