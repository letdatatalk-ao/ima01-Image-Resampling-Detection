"""Moteur de correlation : justesse et vitesse (rapport Section 4.2, Table 2).

Trois methodes, meme backend (numpy), meme sortie (n_patches, dmax+1), axe vertical :
    naive     : formule directe (src.spectral.patch_correlations_naive)  -> reference
    nous      : FFT + sommes glissantes (src.spectral.patch_correlations_fast)
    integral  : portage numpy du code des auteurs, FFT + images integrales (paper_corr_numpy.py)
Chrono : la FFT image -> spectre est incluse pour les trois ; 1 appel de chauffe, puis la mediane.
Usage (depuis code_ima) : python test_contribution/benchmark.py
"""
import os, sys, time, statistics
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np
from src.spectral import patch_correlations_naive, patch_correlations_fast
from paper_corr_numpy import compute_corr_distances_np


def timed(fn, repeats=5, warmup=1):
    """Mediane du temps de fn() sur `repeats` essais, apres `warmup` appels jetes."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter(); fn(); times.append(time.perf_counter() - t0)
    return statistics.median(times)


def run(N, ratio, dmax, with_naive, seed=0):
    img = np.random.default_rng(seed).standard_normal((N, N))
    h = w = round(ratio * N)
    spec = np.fft.fft2(img)
    ours = patch_correlations_fast(spec, (h, w), dmax, axis=0)
    integ = compute_corr_distances_np(img, ratio, dmax).T
    ref = patch_correlations_naive(spec, (h, w), dmax, axis=0) if with_naive else ours   # naive trop lent au-dela de 256
    err = {"nous": np.max(np.abs(ours - ref)) if with_naive else None, "integral": np.max(np.abs(integ - ref))}
    t = {"nous": timed(lambda: patch_correlations_fast(np.fft.fft2(img), (h, w), dmax, axis=0)),
         "integral": timed(lambda: compute_corr_distances_np(img, ratio, dmax))}
    if with_naive:
        t["naive"] = timed(lambda: patch_correlations_naive(np.fft.fft2(img), (h, w), dmax, axis=0), repeats=3)
    return N, ours.shape[0], dmax, err, t


rows = [run(*c) for c in [(128, 0.1, 64, True), (256, 0.1, 128, True), (512, 0.1, 128, False), (768, 0.1, 128, False)]]
print(f"{'N':>5} {'patches':>8} {'dmax':>5} | {'erreur nous':>12} {'erreur integral':>16} | {'naive':>8} {'nous':>8} {'integral':>9}")
for N, n_p, dmax, err, t in rows:
    e_ours = f"{err['nous']:.1e}" if err["nous"] is not None else "reference"
    t_naive = f"{t['naive']:.3f}" if "naive" in t else "--"
    print(f"{N:>5} {n_p:>8} {dmax:>5} | {e_ours:>12} {err['integral']:>16.1e} | {t_naive:>8} {t['nous']:>8.4f} {t['integral']:>9.4f}")
