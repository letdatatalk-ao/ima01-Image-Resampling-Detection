import numpy as np
from src.detector import detect, matches
from src.resampling import resample_2d_fast, cubic_kernel


def test_detect_bruit_reechantillonne_au_bon_d():
    rng=np.random.default_rng(0)
    v=resample_2d_fast(rng.integers(0,256,(100,100)).astype(float),(130,130),cubic_kernel)
    o=detect(v)
    assert o["detected"] and matches(list(o["d_v"]),[30,100])


def test_pas_de_detection_sur_bruit_original():
    o=detect(np.random.default_rng(1).integers(0,256,(130,130)).astype(float))
    assert not o["detected"]


def test_matches_tolerance():
    assert matches([101],[100.0]) and not matches([103],[100.0])