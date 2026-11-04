"""Lecture d'images reelles (PNG, JPEG, TIFF 8/16 bits) et conversion en luminance.
Convention : toutes les images sont rendues en float64 sur l'echelle [0, 255],
pour etre compatibles avec le rank transform et le TV (lambda=1 sur [0,255])."""
import os
import numpy as np
import imageio.v3 as iio
from PIL import Image

# Coefficients ITU-R BT.601 (luminance "Y" classique, aussi utilisee par PIL en mode "L")
LUMA_BT601=(0.299,0.587,0.114)


def _imread(path):
    """imageio d'abord ; si le TIFF est compresse (LZW, Deflate...) et qu'imagecodecs
    n'est pas installe, on passe par Pillow, qui sait decoder ces formats."""
    try:
        return iio.imread(path)
    except Exception as e:
        try:
            with Image.open(path) as im:
                return np.asarray(im)
        except Exception:
            raise e


def read_image(path):
    """Lit une image et la renvoie en float64 sur [0,255].
    - niveaux de gris : (H,W) ; couleur : (H,W,3) ; le canal alpha est supprime.
    - les images 16 bits (TIFF) sont ramenees a [0,255]."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    img=_imread(path)
    if img.ndim==3 and img.shape[2]==4:        # RGBA -> RGB
        img=img[:,:,:3]
    if img.ndim==3 and img.shape[2]==1:
        img=img[:,:,0]
    if img.ndim not in (2,3):
        raise ValueError(f"forme d'image non geree : {img.shape}")
    if img.dtype==np.uint8:
        out=img.astype(np.float64)
    elif img.dtype==np.uint16:
        out=img.astype(np.float64)*(255.0/65535.0)
    elif np.issubdtype(img.dtype,np.floating):
        out=img.astype(np.float64)
        if out.max()<=1.0:                    # image flottante sur [0,1]
            out=out*255.0
    else:
        raise ValueError(f"type de pixel non gere : {img.dtype}")
    return out


def rgb_to_luminance(img,coeffs=LUMA_BT601):
    """(H,W,3) -> (H,W) par Y = 0.299 R + 0.587 G + 0.114 B. Une image 2D est renvoyee telle quelle."""
    img=np.asarray(img,dtype=np.float64)
    if img.ndim==2:
        return img.copy()
    if img.ndim!=3 or img.shape[2]!=3:
        raise ValueError(f"image RGB (H,W,3) attendue, recu {img.shape}")
    r,g,b=coeffs
    return r*img[:,:,0]+g*img[:,:,1]+b*img[:,:,2]


def load_gray(path):
    """Lecture + luminance : c'est l'entree du detecteur."""
    return rgb_to_luminance(read_image(path))


def save_gray_png(img,path):
    """Sauvegarde une image en niveaux de gris 8 bits (arrondi + saturation dans [0,255])."""
    img=np.asarray(img,dtype=np.float64)
    if img.ndim!=2:
        raise ValueError("image 2D attendue")
    os.makedirs(os.path.dirname(os.path.abspath(path)),exist_ok=True)
    iio.imwrite(path,np.clip(np.round(img),0,255).astype(np.uint8))


def image_info(path):
    """Taille, type et nombre de canaux, sans conversion (utile pour verifier un telechargement)."""
    img=_imread(path)
    return {"shape":img.shape,"dtype":str(img.dtype),"min":float(img.min()),"max":float(img.max())}