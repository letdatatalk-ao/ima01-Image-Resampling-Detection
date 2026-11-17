"""Prepare le jeu de test RAISE (protocole Sec. IV-A du papier) a partir des TIFF telecharges.

Pour chaque photo :  TIFF -> luminance -> x1/2 (moyenne 2x2) -> [reechantillonnage x q, 4 noyaux] -> centre 600x600
                     -> PNG, JPEG 95, JPEG 90

Organisation (dossiers par type + index) :
    data/raise_prepared/
        index.csv
        original/{png,jpeg95,jpeg90}/<id>.<ext>                      (q = 1, negatifs)
        resampled/<noyau>/q<q>/{png,jpeg95,jpeg90}/<id>.<ext>        (positifs)

Exemples (depuis code_ima) :
    python scripts/prepare_raise.py --n 3                 # essai sur 3 photos
    python scripts/prepare_raise.py --workers 4           # toutes les photos du manifest
Les photos deja preparees (tous leurs fichiers presents) sont sautees : on peut relancer.
"""
import argparse, csv, os, sys, time
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.image_io import load_gray, save_gray_png, save_gray_jpeg
from src.dataset import KERNELS, FACTORS, OUT_SIZE, downsample2, center_crop, resample_and_crop, expected_distances

FORMATS = [("png", "png", None), ("jpeg95", "jpg", 95), ("jpeg90", "jpg", 90)]
COLS = ["id", "device", "label", "kernel", "q", "format", "path", "size", "d1", "d2", "source"]


def outputs_for(out, iid):
    """Liste (condition, chemin) de tous les fichiers a produire pour une photo."""
    res = []
    for fmt, ext, _ in FORMATS:
        res.append((("original", "none", 1.0, fmt), os.path.join(out, "original", fmt, f"{iid}.{ext}")))
    for k in KERNELS:
        for q in FACTORS:
            for fmt, ext, _ in FORMATS:
                res.append((("resampled", k, q, fmt), os.path.join(out, "resampled", k, f"q{q:.1f}", fmt, f"{iid}.{ext}")))
    return res


def save(img, path, fmt):
    q = dict((f, qq) for f, _, qq in FORMATS)[fmt]
    if q is None:
        save_gray_png(img, path)
    else:
        save_gray_jpeg(img, path, quality=q)


def process(job):
    """Traite une photo ; renvoie (id, lignes d'index, message)."""
    row, out = job
    iid, src = row["File"], row["path"]
    todo = outputs_for(out, iid)
    rows = [{"id": iid, "device": row.get("Device", ""), "label": lab, "kernel": k, "q": q, "format": fmt,
             "path": os.path.relpath(p, out), "size": OUT_SIZE,
             "d1": "" if lab == "original" else round(expected_distances(q)[0], 2),
             "d2": "" if lab == "original" else round(expected_distances(q)[1], 2), "source": src}
            for (lab, k, q, fmt), p in todo]
    if all(os.path.exists(p) for _, p in todo):
        return iid, rows, "deja prepare"
    try:
        t0 = time.time()
        half = downsample2(load_gray(src))                     # efface le dematricage
        orig = center_crop(half, OUT_SIZE)
        cache = {}
        for (lab, k, q, fmt), p in todo:
            if os.path.exists(p):
                continue
            if lab == "original":
                img = orig
            else:
                if (k, q) not in cache:
                    cache.clear(); cache[(k, q)] = resample_and_crop(half, q, k)
                img = cache[(k, q)]
            save(img, p, fmt)
        return iid, rows, f"ok ({time.time() - t0:.1f}s)"
    except Exception as e:
        return iid, [], f"ECHEC : {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/raise/tiff/manifest.csv")
    ap.add_argument("--out", default="data/raise_prepared")
    ap.add_argument("--n", type=int, default=None, help="nombre max de photos (defaut : toutes)")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()

    with open(a.manifest, newline="", encoding="utf-8") as f:
        photos = [r for r in csv.DictReader(f) if os.path.exists(r["path"])]
    photos = photos[:a.n] if a.n else photos
    if not photos:
        sys.exit("aucune photo trouvee : lancez d'abord scripts/download_raise.py")
    per = len(outputs_for(a.out, "x"))
    print(f"{len(photos)} photo(s) x {per} fichiers = {len(photos) * per} images de test -> {a.out}")

    all_rows, failed = [], []
    jobs = [(r, a.out) for r in photos]
    it = Pool(a.workers).imap(process, jobs) if a.workers > 1 else map(process, jobs)
    for i, (iid, rows, msg) in enumerate(it, 1):
        print(f"[{i}/{len(photos)}] {iid} : {msg}", flush=True)
        (failed.append(iid) if msg.startswith("ECHEC") else all_rows.extend(rows))

    idx = os.path.join(a.out, "index.csv")
    old = {}
    if os.path.exists(idx):
        with open(idx, newline="", encoding="utf-8") as f:
            old = {r["path"]: r for r in csv.DictReader(f)}
    old.update({r["path"]: r for r in all_rows})
    os.makedirs(a.out, exist_ok=True)
    with open(idx, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(old.values())
    print(f"\nindex : {idx} ({len(old)} lignes) ; echecs : {len(failed)} {' '.join(failed)}")


if __name__ == "__main__":
    main()