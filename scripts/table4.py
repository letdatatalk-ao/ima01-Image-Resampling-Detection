"""Taux de detection (%) a 1 % de faux positifs, noyau bicubique (rapport Tables 4 et 5).

Pour chaque format (PNG, JPEG 95, JPEG 90) :
  1. seuil calibre sur les ORIGINALES : t = valeur de log10 NFA de rang floor(0.01 n),
     une image est detectee si log10 NFA < t (donc au plus 1 % d'originales detectees) ;
  2. taux de detection des images reechantillonnees, pour chaque facteur q ;
  3. avec validation croisee (colonne log10_nfa) et sans (log10_nfa_nocv).
Usage (depuis code_ima) :  python scripts/table4.py [--dir results/final] [--kernel cubic]
"""
import argparse, os
import numpy as np
import pandas as pd

QS = [0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
# valeurs publiees par les auteurs (Table IV, noyau bicubique) ; None = facteur ignore en JPEG
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


def fmt_row(name, values):
    cells = "".join(f"{x:7.1f}" if x is not None and not np.isnan(x) else "      -" for x in values)
    ok = [x for x in values if x is not None and not np.isnan(x)]
    return f"     {name:7s}{cells}   {np.mean(ok):6.1f}"


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
        orig = r[r["label"] == "original"]
        res = r[(r["label"] == "resampled") & (r["kernel"] == a.kernel)]
        fp = 100 * (orig["detected"].astype(str) == "True").mean()
        print(f"\n=== {fmt.upper()}  ({len(orig)} originales)  faux positifs a eps=1 : {fp:.1f} %")
        for mode, col in [("cv", "log10_nfa"), ("nocv", "log10_nfa_nocv")]:
            t = threshold_1pct(orig[col])
            acc = [100 * (res[res["q"] == q][col] < t).mean() if (res["q"] == q).any() else np.nan for q in QS]
            if fmt != "png":                         # 0.8 et 1.6 tombent sur la grille JPEG kN/8
                acc = [np.nan if p_ is None else x for x, p_ in zip(acc, PAPER[fmt][mode])]
            print(f"  {'avec' if mode == 'cv' else 'sans'} validation croisee, seuil log10 NFA < {t:.2f}")
            print("     q      " + "".join(f"{q:>7.1f}" for q in QS) + "   moyenne")
            print(fmt_row("nous", acc)); print(fmt_row("papier", PAPER[fmt][mode]))
            for q, x, y in zip(QS, acc, PAPER[fmt][mode]):
                rows.append({"format": fmt, "mode": mode, "q": q, "ours": None if np.isnan(x) else round(x, 1),
                             "paper": y, "threshold_log10": round(t, 3), "n_originals": len(orig)})
    if rows:
        out = os.path.join(a.dir, "table4.csv"); pd.DataFrame(rows).to_csv(out, index=False); print(f"\n-> {out}")


if __name__ == "__main__":
    main()
