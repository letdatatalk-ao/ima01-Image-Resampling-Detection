import numpy as np
from src.resampling import resample_2d_sep, resample_2d_fast
from src.dataset import downsample2, center_crop, crop_size_for, resample_and_crop, expected_distances, FACTORS, KERNELS


def test_resample_fast_egal_a_la_version_de_reference():
    u=np.random.default_rng(0).random((40,30))
    for k in KERNELS.values():
        assert np.allclose(resample_2d_fast(u,(57,23),k),resample_2d_sep(u,(57,23),k),atol=1e-12)


def test_q1_ne_change_pas_l_image():
    """Noyaux interpolants : phi(0)=1, phi(k)=0 -> un reechantillonnage x1 rend l'image exacte."""
    u=np.random.default_rng(1).random((30,30))
    for k in KERNELS.values():
        assert np.allclose(resample_2d_fast(u,(30,30),k),u,atol=1e-12)


def test_downsample2_moyenne_2x2():
    x=np.arange(16,dtype=float).reshape(4,4)
    assert np.allclose(downsample2(x),[[2.5,4.5],[10.5,12.5]])


def test_tailles_exactes():
    for q in FACTORS:
        s_in,s_out=crop_size_for(q)
        assert abs(s_out/s_in-q)<1e-12 and s_out>=600+16


def test_sortie_600x600():
    img=np.random.default_rng(2).random((1200,1100))*255
    for q in (0.6,1.6):
        assert resample_and_crop(img,q,"cubic").shape==(600,600)


def test_distances_attendues():
    assert expected_distances(1.2)==[100.0,500.0]
    assert expected_distances(0.8)==[150.0,450.0]
