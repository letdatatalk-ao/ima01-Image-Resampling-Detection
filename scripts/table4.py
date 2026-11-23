"""Reproduction de la Table IV du papier : precision de detection (%) a 1 % de faux positifs.

Pour chaque format (PNG, JPEG 95, JPEG 90) :
  1. le seuil est calibre sur les ORIGINALES : on garde 1 % de faux positifs
     (seuil t = la valeur de log10 NFA de rang floor(0.01 n), detection si log10 NFA < t) ;
  2. on mesure le taux de detection des images reechantillonnees (noyau bicubique par defaut) par facteur q ;
  3. avec validation croisee (log10_nfa) et sans (log10_nfa_nocv), comme les deux lignes du papier.

Usage (depuis code_ima) :  python scripts/table4.py [--kernel cubic] [--dir results/final]
Sorties : results/final/table4.csv et un tableau affiche avec les valeurs du papier.
"""
import argparse, os, sys
import numpy as np
import pandas as pd

QS = [0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
# Table IV du papier (bicubique) : "Ours" (sans validation croisee) et "Ours, cross-val."
PAPER = {
    "png":    {"nocv": [93.1, 99.9, 100.0, 99.8, 99.7, 99.9, 99.8, 99.8, 99.8, 99.8],
               "cv":   [98.1, 100.0, 99.9, 99.9, 100.0, 99.9, 100.0, 99.9, 100.0, 100.0]},
    "jpeg95": {"nocv": [48.0, 78.1, None, 93.0, 98.0, 98.9, 97.5, 98.1, 99.6, None],
               "cv":   [60.0, 88.4, None, 93.0, 97.5, 99.1, 97.5, 98.6, 99.6, None]},
    "jpeg90": {"nocv": [26.6, 41.3, None, 66.8, 88.2, 93.1, 83.9, 86.9, 96.5, None],
               "cv":   [36.0, 55.8, None, 66.1, 89.0, 93.0, 88.6, 90.6, 96.1, None]},
}
FILES = {"png": "detection_png_rt.csv", "jpeg95": "detection_jpeg95_tv.csv", "jpeg90": "detection_jpeg90_tv.csv"}


def threshold_1pct(scores):
    """Seuil t tel qu'au plus 1 % des originales aient un score < t."""
    s = np.sort(np.asarray(scores, dtype=float))
    return s[int(np.floor(0.01 * len(s)))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="results/final")
    ap.add_argument("--kernel", default="cubic")
    a = ap.parse_args()
    rows = []
    for fmt, fname in FILES.items():
        p = os.path.join(a.dir, fname)
        if not os.path.exists(p):
            print(f"[{fmt}] {p} absent : lancez run_detection.py --format {fmt}")
            continue
        r = pd.read_csv(p)
        r = r[r["detected"].astype(str) != "ERROR"]
        o = r[r["label"] == "original"]; s = r[(r["label"] == "resampled") & (r["kernel"] == a.kernel)]
        n0 = len(o)
        fp_eps1 = 100 * (o["detected"].astype(str) == "True").mean()
        print(f"\n=== {fmt.upper()}  ({n0} originales, {s['id'].nunique()} photos)  faux positifs a eps=1 : {fp_eps1:.1f} %")
        print("  faux positifs des originales selon eps (calibration H0 en 2D) :")
        for col, lab in [("log10_nfa", "notre c.v."), ("log10_nfa_h", "H seul    ")]:
            if col in o:
                print("     " + lab + "  " + "  ".join(f"eps=1e{e}: {100 * (o[col] < e).mean():5.1f}%" for e in (0, -1, -2, -3)))
        variants = [("nocv", "log10_nfa_h", "H seul, sans c.v. (= ligne 'Ours')"),
                    ("cv", "log10_nfa_cv_authors", "c.v. des auteurs (= 'Ours, cross-val.')"),
                    ("cv", "log10_nfa", "notre c.v. symetrique"),
                    ("nocv", "log10_nfa_nocv", "2 axes sans c.v.")]
        for mode, col, label in variants:
            if col not in r:
                print(f"  colonne {col} absente (relancez run_detection.py)"); continue
            t = threshold_1pct(o[col])
            acc = [100 * (s[s["q"] == q][col] < t).mean() if (s["q"] == q).any() else np.nan for q in QS]
            pap = PAPER[fmt][mode]
            print(f"  {label} -- seuil 1% : log10 NFA < {t:.2f}  (eps = {10 ** t:.1e})")
            print("     q      " + "".join(f"{q:>7.1f}" for q in QS) + "   moyenne")
            ok = [x for x in acc if not np.isnan(x)]
            print("     nous   " + "".join(f"{x:7.1f}" if not np.isnan(x) else "      -" for x in acc) + f"   {np.mean(ok):6.1f}")
            pk = [x for x in pap if x is not None]
            print("     papier " + "".join(f"{x:7.1f}" if x is not None else "      -" for x in pap) + f"   {np.mean(pk):6.1f}")
            for q, x, y in zip(QS, acc, pap):
                rows.append({"format": fmt, "variant": col, "mode": mode, "q": q, "ours": None if np.isnan(x) else round(x, 1), "paper": y,
                             "threshold_log10": round(t, 3), "n_originals": n0})
    if rows:
        out = os.path.join(a.dir, "table4.csv"); pd.DataFrame(rows).to_csv(out, index=False); print(f"\n-> {out}")
    print("\nNote : en JPEG, le papier ignore q=0.8 et q=1.6 : 600/0.8 mod 600 = 150 et 600/1.6 = 375 tombent sur des distances JPEG kN/8.")


if __name__ == "__main__":
    main()