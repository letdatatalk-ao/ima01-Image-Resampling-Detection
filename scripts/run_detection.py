"""Lance le detecteur sur la dataset preparee (index.csv) et resume les resultats.

Exemples (depuis code_ima) :
    python scripts/run_detection.py                          # tous les PNG
    python scripts/run_detection.py --kernel cubic --workers 4
    python scripts/run_detection.py --format jpeg90 --s 1    # plus tard, apres l'exclusion JPEG

Sorties :
    results/detection_<format>.csv : une ligne par image (decision, distances, NFA, bon d ?)
    resume affiche : faux positifs sur les originales, taux de detection par noyau et facteur q
Les images deja traitees (presentes dans le CSV de resultats) sont sautees.
"""
import argparse, csv, os, sys, time
from multiprocessing import Pool
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.detector import detect, matches, harmonic_distances

COLS = ["id", "label", "kernel", "q", "format", "path", "detected", "correct_d", "correct_d_harm", "log10_nfa", "log10_nfa_nocv",
        "d1", "d2", "d_v", "d_h", "best_v", "best_h", "seconds"]


def run_one(job):
    row, root, params = job
    t0 = time.time()
    try:
        o = detect(os.path.join(root, row["path"]), **params)
    except Exception as e:
        return {**row, "detected": "ERROR", "correct_d": "", "log10_nfa": "", "d_v": str(e), "d_h": "",
                "best_v": "", "best_h": "", "seconds": ""}
    exp = [float(row["d1"]), float(row["d2"])] if row["label"] == "resampled" else []
    dv, dh = [int(x) for x in o["d_v"]], [int(x) for x in o["d_h"]]
    ok = (matches(dv, exp) or matches(dh, exp)) if exp else ""
    harm = harmonic_distances(float(row["q"]), o["size"][0]) if exp else []
    okh = (matches(dv, harm) or matches(dh, harm)) if exp else ""
    return {**{k: row[k] for k in ["id", "label", "kernel", "q", "format", "path", "d1", "d2"]},
            "detected": o["detected"], "correct_d": ok, "correct_d_harm": okh,
            "log10_nfa": round(o["log10_nfa"], 2), "log10_nfa_nocv": round(o["log10_nfa_nocv"], 2),
            "d_v": " ".join(map(str, dv)), "d_h": " ".join(map(str, dh)),
            "best_v": o["best_v"][0], "best_h": o["best_h"][0], "seconds": round(time.time() - t0, 2)}


def summary(rows):
    T = lambda x: str(x) == "True"
    orig = [r for r in rows if r["label"] == "original"]
    print("\n================  RESUME  ================")
    if orig:
        fp = sum(T(r["detected"]) for r in orig)
        print(f"Originales : {fp}/{len(orig)} faux positifs ({100 * fp / len(orig):.1f} %)")
    res = defaultdict(list)
    for r in rows:
        if r["label"] == "resampled":
            res[(r["kernel"], float(r["q"]))].append(r)
    if not res:
        return
    kernels = sorted({k for k, _ in res}); qs = sorted({q for _, q in res})
    for title, key in [("Taux de detection (%)", "detected"), ("Detection au BON d (%)", "correct_d")]:
        print(f"\n{title}")
        print("noyau    " + "".join(f"{q:>7.1f}" for q in qs))
        for k in kernels:
            line = f"{k:8s} "
            for q in qs:
                g = res.get((k, q), [])
                line += f"{100 * sum(T(r[key]) for r in g) / len(g):7.0f}" if g else "      -"
            print(line)
    n = sum(len(g) for g in res.values())
    print(f"\n({n} images reechantillonnees, {len({r['id'] for r in rows})} photos)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/raise_prepared")
    ap.add_argument("--format", default="png", choices=["png", "jpeg95", "jpeg90"])
    ap.add_argument("--kernel", default=None, help="un seul noyau (defaut : tous)")
    ap.add_argument("--residual", default=None, choices=["rt", "tv", "none"],
                    help="defaut : rt (7x7) pour PNG, tv pour JPEG, comme le papier (Sec. IV)")
    ap.add_argument("--s", type=int, default=None, help="demi-fenetre du rank transform (defaut 3 en PNG, 1 en JPEG)")
    ap.add_argument("--no-jpeg-suppression", action="store_true", help="desactive la suppression des pics kN/8 (Sec. III-C)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    with open(os.path.join(a.data, "index.csv"), newline="", encoding="utf-8") as f:
        idx = [r for r in csv.DictReader(f) if r["format"] == a.format
               and (a.kernel is None or r["label"] == "original" or r["kernel"] == a.kernel)]
    is_jpeg = a.format.startswith("jpeg")
    a.residual = a.residual or ("tv" if is_jpeg else "rt")
    a.s = a.s if a.s is not None else (1 if is_jpeg else 3)
    params = {"residual": a.residual, "s": a.s, "jpeg": is_jpeg and not a.no_jpeg_suppression}
    out = a.out or f"results/final/detection_{a.format}_{a.residual}{'' if params['jpeg'] or not is_jpeg else '_nosupp'}.csv"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    done = {}
    if os.path.exists(out):
        with open(out, newline="", encoding="utf-8") as f:
            done = {r["path"]: r for r in csv.DictReader(f)}
    todo = [r for r in idx if r["path"] not in done]
    print(f"{len(idx)} images ({a.format}), {len(done)} deja faites, {len(todo)} a traiter -> {out}")

    t0 = time.time()
    jobs = [(r, a.data, params) for r in todo]
    it = Pool(a.workers).imap(run_one, jobs) if a.workers > 1 else map(run_one, jobs)
    for i, res in enumerate(it, 1):
        done[res["path"]] = res
        if i % 20 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}  ({time.time() - t0:.0f}s)", flush=True)
            with open(out, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore"); w.writeheader(); w.writerows(done.values())
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore"); w.writeheader(); w.writerows(done.values())
    summary(list(done.values()))
    print(f"\nresultats : {out}")


if __name__ == "__main__":
    main()