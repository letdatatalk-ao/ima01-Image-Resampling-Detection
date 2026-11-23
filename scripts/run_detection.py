"""Lance le detecteur sur le jeu RAISE prepare (index.csv), ecrit un CSV et affiche un resume.

Exemples (depuis code_ima) :
    python scripts/run_detection.py --workers 4                      # PNG, rank transform 7x7
    python scripts/run_detection.py --format jpeg90 --workers 4      # JPEG : residu TV + suppression kN/8
    python scripts/run_detection.py --kernel cubic --residual none   # ablation du residu

Sortie : results/final/detection_<format>_<residu>.csv, une ligne par image :
    detected (NFA < 1), correct_d (une distance detectee = la distance attendue +-1),
    log10_nfa (avec validation croisee), log10_nfa_nocv (sans), d_v / d_h (distances detectees).
Les images deja presentes dans le CSV sont sautees : on peut relancer apres une coupure.
"""
import argparse, csv, os, sys, time
from multiprocessing import Pool
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.detector import detect, matches, harmonic_distances

COLS = ["id", "label", "kernel", "q", "format", "path", "detected", "correct_d", "correct_d_harm", "log10_nfa", "log10_nfa_nocv",
        "d1", "d2", "d_v", "d_h", "best_v", "best_h", "seconds"]


def run_one(job):
    """Detecte une image et compare aux distances attendues d1, d2 (= +-600/q mod 600)."""
    row, root, params = job
    t0 = time.time()
    try:
        o = detect(os.path.join(root, row["path"]), **params)
    except Exception as e:
        return {**row, "detected": "ERROR", "d_v": str(e)}
    expected = [float(row["d1"]), float(row["d2"])] if row["label"] == "resampled" else []
    dv, dh = [int(x) for x in o["d_v"]], [int(x) for x in o["d_h"]]
    ok = (matches(dv, expected) or matches(dh, expected)) if expected else ""
    harm = harmonic_distances(float(row["q"]), o["size"][0]) if expected else []
    ok_harm = (matches(dv, harm) or matches(dh, harm)) if expected else ""
    return {**{k: row[k] for k in ["id", "label", "kernel", "q", "format", "path", "d1", "d2"]},
            "detected": o["detected"], "correct_d": ok, "correct_d_harm": ok_harm,
            "log10_nfa": round(o["log10_nfa"], 2), "log10_nfa_nocv": round(o["log10_nfa_nocv"], 2),
            "d_v": " ".join(map(str, dv)), "d_h": " ".join(map(str, dh)),
            "best_v": o["best_v"][0], "best_h": o["best_h"][0], "seconds": round(time.time() - t0, 2)}


def summary(rows):
    """Faux positifs sur les originales, puis taux de detection par noyau et par facteur q."""
    is_true = lambda x: str(x) == "True"
    orig = [r for r in rows if r["label"] == "original"]
    print("\n================  RESUME  ================")
    if orig:
        fp = sum(is_true(r["detected"]) for r in orig)
        print(f"Originales : {fp}/{len(orig)} faux positifs ({100 * fp / len(orig):.1f} %)")
    groups = defaultdict(list)
    for r in rows:
        if r["label"] == "resampled":
            groups[(r["kernel"], float(r["q"]))].append(r)
    if not groups:
        return
    kernels = sorted({k for k, _ in groups}); qs = sorted({q for _, q in groups})
    for title, key in [("Taux de detection (%)", "detected"), ("Detection au bon d (%)", "correct_d")]:
        print(f"\n{title}")
        print("noyau    " + "".join(f"{q:>7.1f}" for q in qs))
        for k in kernels:
            line = f"{k:8s} "
            for q in qs:
                g = groups.get((k, q), [])
                line += f"{100 * sum(is_true(r[key]) for r in g) / len(g):7.0f}" if g else "      -"
            print(line)


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/raise_prepared")
    ap.add_argument("--format", default="png", choices=["png", "jpeg95", "jpeg90"])
    ap.add_argument("--kernel", default=None, help="un seul noyau (defaut : les quatre)")
    ap.add_argument("--residual", default=None, choices=["rt", "tv", "none"], help="defaut : rt en PNG, tv en JPEG (comme le papier)")
    ap.add_argument("--s", type=int, default=3, help="demi-fenetre du rank transform (3 -> 7x7)")
    ap.add_argument("--no-jpeg-suppression", action="store_true", help="les pics kN/8 votent quand meme")
    ap.add_argument("--jpeg-block", type=int, default=8, help="distances exclues kN/block (8 = papier, 16 = code des auteurs)")
    ap.add_argument("--hermitien", action="store_true", help="un patch par paire miroir (option des limites)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    with open(os.path.join(a.data, "index.csv"), newline="", encoding="utf-8") as f:
        index = [r for r in csv.DictReader(f) if r["format"] == a.format
                 and (a.kernel is None or r["label"] == "original" or r["kernel"] == a.kernel)]
    is_jpeg = a.format.startswith("jpeg")
    residual = a.residual or ("tv" if is_jpeg else "rt")
    params = {"residual": residual, "s": a.s, "jpeg": is_jpeg and not a.no_jpeg_suppression,
              "jpeg_block": a.jpeg_block, "hermitien": a.hermitien}
    base = "results/hermitien" if a.hermitien else "results/final"
    suffix = ("_nosupp" if is_jpeg and not params["jpeg"] else "") + (f"_b{a.jpeg_block}" if a.jpeg_block != 8 else "")
    out = a.out or f"{base}/detection_{a.format}_{residual}{suffix}.csv"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    done = {}
    if os.path.exists(out):
        with open(out, newline="", encoding="utf-8") as f:
            done = {r["path"]: r for r in csv.DictReader(f)}
    todo = [r for r in index if r["path"] not in done]
    print(f"{len(index)} images ({a.format}), {len(done)} deja faites, {len(todo)} a traiter -> {out}")

    t0 = time.time()
    jobs = [(r, a.data, params) for r in todo]
    results = Pool(a.workers).imap(run_one, jobs) if a.workers > 1 else map(run_one, jobs)
    for i, res in enumerate(results, 1):
        done[res["path"]] = res
        if i % 20 == 0 or i == len(todo):            # sauvegarde reguliere
            print(f"  {i}/{len(todo)}  ({time.time() - t0:.0f}s)", flush=True)
            write_csv(out, done.values())
    write_csv(out, done.values())
    summary(list(done.values()))
    print(f"\nresultats : {out}")


if __name__ == "__main__":
    main()
