import numpy as np
from src.resampling import resample_1d, kernel_nn
from src.dft import dft
from src.spectral import patch_correlations_1d
from src.acontrario import compute_nfa, decide


def test_vrai_positif_detecte_M_mod_N():
    """Bruit blanc rééchantillonné"""
    rng = np.random.default_rng(0)
    M,N,L,r = 100, 250, 10, 20
    dmax=N - 1
    u=rng.standard_normal(M)
    v=resample_1d(u,N,kernel_nn)
    rho=patch_correlations_1d(dft(v),L,dmax)
    nfa=compute_nfa(rho,r)
    detections = decide(nfa, eps=1.0)
    assert (M % N) in detections


def test_vrai_negatif_avec_seuil_resserre():
    """Bruit blanc NON rééchantillonné"""
    rng = np.random.default_rng(0)
    M,N,L,r = 100,250,10,20
    dmax= N-1
    w= rng.standard_normal(N)
    rho_ref = patch_correlations_1d(dft(w), L, dmax)
    nfa_ref = compute_nfa(rho_ref, r)
    detections = decide(nfa_ref, eps=1e-3)
    assert len(detections) == 0

def test_jpeg_distances_600():
    from src.acontrario import jpeg_distances
    assert jpeg_distances(600)==[75,150,225,300,375,450,525]


def test_suppression_jpeg_invalide_le_pic_et_son_voisinage():
    """Un pic a d*=75 est invalide jusqu'au 2e minimum local de chaque cote (Sec. III-C)."""
    from src.acontrario import jpeg_valid_mask
    x=np.random.default_rng(0).random(200)*0.1; x[75]=1.0
    v=jpeg_valid_mask(x,[75])
    assert not v[75] and v[0] and v[199]
    assert v.sum()<200-3


def test_votes_sans_exclusion_inchanges():
    from src.acontrario import count_votes
    rho=np.random.default_rng(1).random((30,120))
    assert np.array_equal(count_votes(rho,10,None),count_votes(rho,10,[]))