"""Donnees de la video de soutenance (presentation_10min.py) -> video_cache.json.

Tout vient de NOTRE code (src/) et de NOS resultats (results/final/*.csv) :
  - demo1d : l'exemple fil rouge 1D (meme cache que avancement_1D.py)
  - spec2d : spectre 2D d'un bruit reechantillonne (paysage 3D de la section "theoreme 1")
  - baboon : image de test des auteurs (data/ird_img), originale et agrandie 512 -> 666
  - raise  : statistiques des 200 photos RAISE pour chaque format deja calcule

Usage (depuis code_ima) :  python video_data.py      # a relancer apres chaque run_detection.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.acontrario import compute_nfa, count_votes, detect_2d  # noqa: E402
from src.image_io import load_gray  # noqa: E402
from src.residual import extract_residual  # noqa: E402
from src.resampling import linear_kernel, resample_2d_fast  # noqa: E402
from src.spectral import hermitian_half, patch_correlations_2d  # noqa: E402

CACHE = HERE / "video_cache.json"
QS = [0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
KERNELS = ["nearest", "linear", "cubic", "lanczos"]
FILES = {"png": "detection_png_rt.csv", "jpeg95": "detection_jpeg95_tv.csv", "jpeg90": "detection_jpeg90_tv.csv"}
# Table IV du papier (bicubique, 1 % de faux positifs) : "Ours, cross-val." et "Ours"
PAPER_CV = {"png": [98.1, 100.0, 99.9, 99.9, 100.0, 99.9, 100.0, 99.9, 100.0, 100.0],
            "jpeg95": [60.0, 88.4, None, 93.0, 97.5, 99.1, 97.5, 98.6, 99.6, None],
            "jpeg90": [36.0, 55.8, None, 66.1, 89.0, 93.0, 88.6, 90.6, 96.1, None]}


def lg(x):
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore"):
        y = np.log10(x)
    return [None if not np.isfinite(v) else round(float(v), 3) for v in y]


def spec2d():
    """Bruit 30x30 -> 48x48 (lineaire). |DFT| : copies decalees de d = 30 (= -18 mod 48) sur chaque axe."""
    rng = np.random.default_rng(1)
    u = rng.standard_normal((30, 30))
    v = resample_2d_fast(u, (48, 48), linear_kernel)
    A = np.abs(np.fft.fft2(v - v.mean())) / v.size
    A = A / np.percentile(A, 99)
    return {"M": 30, "N": 48, "absV": np.round(np.clip(A, 0, 1.2), 4).tolist()}


def analyse(path, residual="rt", s=3, r=20):
    img = load_gray(str(path))
    R = extract_residual(img, method=residual, s=s)
    H, W = R.shape
    rho_v, rho_h, hw = patch_correlations_2d(np.fft.fft2(R - R.mean()), patch_ratio=10)
    rho_v, rho_h = hermitian_half(rho_v, rho_h, (H, W), hw)   # un patch par paire conjuguee
    out = detect_2d(rho_v, rho_h, H, W, r=r)
    k_h = count_votes(rho_h, r)
    return {"H": H, "W": W, "n_patches": int(rho_h.shape[0]), "k_h": k_h.tolist(),
            "lognfa_h": lg(compute_nfa(rho_h, r)), "lognfa_h_cv": lg(out["nfa_h_cv"]),
            "d_h": [int(d) for d in out["d_h"]], "d_v": [int(d) for d in out["d_v"]],
            "best_h": [out["best_h"][0], lg([out["best_h"][1]])[0]], "detected": out["detected"]}


def baboon():
    d = HERE / "data" / "ird_img"
    if not (d / "baboon.png").exists():
        return None
    res = {"orig": analyse(d / "baboon.png"), "resized": analyse(d / "baboon_666.png")}
    M, N = res["orig"]["W"], res["resized"]["W"]
    res["expected"] = sorted({M % N, (-M) % N})
    return res


def images():
    """Vignettes pour la video : baboon (gris), baboon agrandie, residu rank transform (crop 200x200)."""
    from PIL import Image
    from src.residual import rank_transform_2d
    d = HERE / "data" / "ird_img"
    out = HERE / "video_assets"
    out.mkdir(exist_ok=True)
    if not (d / "baboon.png").exists():
        return False
    a = load_gray(str(d / "baboon.png"))
    b = load_gray(str(d / "baboon_666.png"))
    Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(out / "baboon_gray.png")
    Image.fromarray(np.clip(b, 0, 255).astype(np.uint8)).save(out / "baboon666_gray.png")
    c = a[156:356, 156:356]
    Image.fromarray(np.clip(c, 0, 255).astype(np.uint8)).save(out / "crop_gray.png")
    rt = rank_transform_2d(c, s=3)
    Image.fromarray((255 * rt / 48).astype(np.uint8)).save(out / "crop_rt.png")
    return True


def threshold_1pct(scores):
    s = np.sort(np.asarray(scores, dtype=float))
    return float(s[int(np.floor(0.01 * len(s)))])


def results_dir(name=None):
    """results/corrige (votes independants, version par defaut) si disponible, sinon results/final (papier)."""
    if name:
        return HERE / "results" / name
    c = HERE / "results" / "corrige"
    return c if (c / FILES["png"]).exists() else HERE / "results" / "final"


def raise_stats(name=None):
    out = {}
    for fmt, fname in FILES.items():
        p = results_dir(name) / fname
        if not p.exists():
            continue
        r = pd.read_csv(p)
        r = r[r["detected"].astype(str) != "ERROR"].copy()
        r["det"] = r["detected"].astype(str) == "True"
        r["ok_d"] = r["correct_d"].astype(str) == "True"
        o, s = r[r["label"] == "original"], r[r["label"] == "resampled"]
        heat = [[round(100 * s[(s.kernel == k) & (s.q == q)]["det"].mean(), 1) if ((s.kernel == k) & (s.q == q)).any() else None
                 for q in QS] for k in KERNELS]
        heat_d = [[round(100 * s[(s.kernel == k) & (s.q == q)]["ok_d"].mean(), 1) if ((s.kernel == k) & (s.q == q)).any() else None
                   for q in QS] for k in KERNELS]
        cub = s[s.kernel == "cubic"]
        t4 = {}
        for col in ("log10_nfa", "log10_nfa_nocv"):
            t = threshold_1pct(o[col])
            t4[col] = {"threshold": round(t, 2),
                       "acc": [round(100 * (cub[cub.q == q][col] < t).mean(), 1) if (cub.q == q).any() else None for q in QS]}
        out[fmt] = {"n_orig": int(len(o)), "n_photos": int(r["id"].nunique()), "n_rows": int(len(r)),
                    "fp_eps1": round(100 * o["det"].mean(), 1), "fp_eps1_nocv": round(100 * (o["log10_nfa_nocv"] < 0).mean(), 1),
                    "min_orig_nocv": round(float(o["log10_nfa_nocv"].min()), 1),
                    "heat": heat, "heat_d": heat_d, "table4": t4, "paper_cv": PAPER_CV[fmt],
                    "sec_per_image": round(float(r["seconds"].mean()), 2) if "seconds" in r else None}
    return out


def extra_stats():
    """Ablation du residu (PNG, bicubique) et redimensionneur PIL avec antialiasing."""
    out = {}
    for res in ("none", "tv", "rt"):
        p = results_dir() / f"detection_png_{res}.csv"
        if not p.exists():
            continue
        r = pd.read_csv(p)
        r = r[r["detected"].astype(str) != "ERROR"]
        det = r["detected"].astype(str) == "True"
        o = r["label"] == "original"
        c = (r["kernel"] == "cubic") & ~o
        out[f"res_{res}"] = {"fp": round(100 * det[o].mean(), 1),
                             "acc": [round(100 * det[c & (r.q == q)].mean(), 1) for q in QS]}
    p = results_dir() / "detection_png_rt_pil.csv"
    if p.exists():
        r = pd.read_csv(p)
        r = r[r["detected"].astype(str) != "ERROR"]
        det = r["detected"].astype(str) == "True"
        c = r["label"] != "original"
        out["pil_cubic"] = [round(100 * det[c & (r.q == q)].mean(), 1) for q in QS]
    return out


def compute():
    from avancement_1D import load_results  # le cache 1D de nos experiences (animation_cache.json)
    R = load_results()
    return {"demo1d": R["demo"], "E1": R["E1"], "E2": R["E2"], "E3": R["E3"], "E4": R["E4"],
            "spec2d": spec2d(), "baboon": baboon(), "images": images(), "raise": raise_stats(), "raise_papier": raise_stats("final"), "results_dir": results_dir().name,
            "extra": extra_stats(), "qs": QS, "kernels": KERNELS}


def load(refresh=False):
    if CACHE.exists() and not refresh:
        return json.loads(CACHE.read_text(encoding="utf-8"))
    res = compute()
    CACHE.write_text(json.dumps(res), encoding="utf-8")
    return res


if __name__ == "__main__":
    res = load(refresh=True)
    for fmt, st in res["raise"].items():
        print(f"{fmt}: {st['n_photos']} photos, FP(eps=1) = {st['fp_eps1']} %, Table IV cv = {st['table4']['log10_nfa']['acc']}")
    if res["baboon"]:
        b = res["baboon"]
        print("baboon : attendu", b["expected"], "| agrandie d_h =", b["resized"]["d_h"], "| originale d_h =", b["orig"]["d_h"])
    print("->", CACHE)
