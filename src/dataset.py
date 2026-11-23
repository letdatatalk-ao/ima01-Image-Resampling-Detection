"""Preparation du jeu de test RAISE (rapport Section 6.3, protocole de la Sec. IV-A du papier).
TIFF -> luminance -> x1/2 (moyenne 2x2) -> (reechantillonnage x q) -> recadrage central 600 x 600."""
import numpy as np
from src.resampling import resample_2d_fast, kernel_nn, linear_kernel, cubic_kernel, lanczos_kernel

KERNELS={"nearest":kernel_nn,"linear":linear_kernel,"cubic":cubic_kernel,"lanczos":lanczos_kernel}
FACTORS=[0.6,0.7,0.8,0.9,1.1,1.2,1.3,1.4,1.5,1.6]    # q=1 = image originale, stockee a part
OUT_SIZE=600
MARGIN=8      # pixels jetes de chaque cote : couvre le support des noyaux (3 pour Lanczos) et le bord periodique


def downsample2(img):
    """Sous-echantillonnage x1/2 par moyenne de blocs 2x2 (efface les traces du dematricage).
    Sa trace spectrale tombe a d = 2N mod N = 0, distance jamais testee."""
    img=np.asarray(img,dtype=np.float64)
    h,w=img.shape[0]//2*2,img.shape[1]//2*2
    x=img[:h,:w]
    return 0.25*(x[0::2,0::2]+x[1::2,0::2]+x[0::2,1::2]+x[1::2,1::2])


def center_crop(img,size):
    """Carre central size x size."""
    h,w=img.shape
    if h<size or w<size:
        raise ValueError(f"image {h}x{w} trop petite pour un recadrage {size}x{size}")
    i,j=(h-size)//2,(w-size)//2
    return img[i:i+size,j:j+size]


def crop_size_for(q,out=OUT_SIZE,margin=MARGIN):
    """Taille S_in du carre a decouper avant de reechantillonner, telle que :
    - S_in*q est un entier (S_in multiple de 10 et q au dixieme) -> rapport exactement q ;
    - apres reechantillonnage il reste au moins out + 2*margin pixels."""
    q10=int(round(q*10))
    s_in=int(np.ceil((out+2*margin)/q/10.0))*10
    return s_in,s_in*q10//10


def resample_and_crop(img,q,kernel,out=OUT_SIZE):
    """Decoupe le centre, reechantillonne x q avec le noyau, recadre le centre en out x out."""
    s_in,s_out=crop_size_for(q,out)
    region=center_crop(img,s_in)
    return center_crop(resample_2d_fast(region,(s_out,s_out),KERNELS[kernel]),out)


def expected_distances(q,N=OUT_SIZE):
    """Distances theoriques du pic dans une image N x N reechantillonnee par q :
    tout se passe comme si M = N/q pixels avaient ete reechantillonnes en N -> d = +-N/q mod N."""
    m=N/q
    return sorted([float(m%N),float((-m)%N)])