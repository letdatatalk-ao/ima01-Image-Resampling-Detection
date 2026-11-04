"""Telecharge un sous-ensemble de RAISE-1k a partir du fichier CSV fourni par le site.

Exemples (depuis le dossier code_ima) :
    python scripts/download_raise.py --csv data/RAISE_1k.csv --n 5
    python scripts/download_raise.py --csv data/RAISE_1k.csv --n 50 --start 0
    python scripts/download_raise.py --csv data/RAISE_1k.csv --ids r000da54ft r001d260dt

- Format TIFF par defaut (RGB, deja demosaique). Une image fait 30 a 50 Mo : commencez petit.
- Les fichiers deja presents sont sautes : on peut relancer apres une coupure.
- Un fichier manifest.csv (id, appareil, taille, chemin) est mis a jour dans le dossier de sortie.
"""
import argparse, csv, os, sys, time, urllib.request, urllib.error

FIELDS=["File","Device","Image Size","Keywords"]


def read_csv(path):
    with open(path,newline="",encoding="utf-8",errors="replace") as f:
        rows=list(csv.DictReader(f))
    if not rows or "File" not in rows[0]:
        raise ValueError("CSV inattendu : colonne 'File' absente")
    return rows


def download(url,dest,retries=3,timeout=60):
    """Telecharge url -> dest via un fichier .part (pas de fichier tronque en cas de coupure)."""
    part=dest+".part"
    for k in range(1,retries+1):
        try:
            req=urllib.request.Request(url,headers={"User-Agent":"ima-raise-downloader"})
            with urllib.request.urlopen(req,timeout=timeout) as r, open(part,"wb") as f:
                total=int(r.headers.get("Content-Length") or 0); got=0; t0=time.time()
                while True:
                    chunk=r.read(1<<20)
                    if not chunk:
                        break
                    f.write(chunk); got+=len(chunk)
                    if total:
                        print(f"\r    {got/2**20:6.1f} / {total/2**20:6.1f} Mo  ({got/2**20/max(time.time()-t0,1e-3):.1f} Mo/s)",end="",flush=True)
            print()
            if total and got!=total:
                raise IOError(f"taille recue {got} != {total}")
            os.replace(part,dest)
            return True
        except (urllib.error.URLError,IOError,TimeoutError) as e:
            print(f"\n    essai {k}/{retries} echoue : {e}")
            if os.path.exists(part):
                os.remove(part)
            time.sleep(2*k)
    return False


def update_manifest(out_dir,entries):
    path=os.path.join(out_dir,"manifest.csv")
    old={}
    if os.path.exists(path):
        with open(path,newline="",encoding="utf-8") as f:
            old={r["File"]:r for r in csv.DictReader(f)}
    old.update({e["File"]:e for e in entries})
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=FIELDS+["path"])
        w.writeheader(); w.writerows(old.values())
    return path


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv",required=True,help="fichier CSV de RAISE-1k")
    ap.add_argument("--out",default="data/raise/tiff",help="dossier de sortie")
    ap.add_argument("--n",type=int,default=5,help="nombre d'images a telecharger")
    ap.add_argument("--start",type=int,default=0,help="indice de la premiere ligne du CSV")
    ap.add_argument("--ids",nargs="*",help="liste d'identifiants (colonne File) a la place de --start/--n")
    ap.add_argument("--format",choices=["TIFF","NEF"],default="TIFF")
    a=ap.parse_args()

    rows=read_csv(a.csv)
    sel=[r for r in rows if r["File"] in set(a.ids)] if a.ids else rows[a.start:a.start+a.n]
    if not sel:
        sys.exit("aucune image selectionnee")
    os.makedirs(a.out,exist_ok=True)
    ext=".TIF" if a.format=="TIFF" else ".NEF"
    done,failed=[],[]
    for i,r in enumerate(sel,1):
        dest=os.path.join(a.out,r["File"]+ext)
        print(f"[{i}/{len(sel)}] {r['File']}  {r['Device']}  {r['Image Size']}")
        if os.path.exists(dest) and os.path.getsize(dest)>0:
            print("    deja present, saute")
        elif not download(r[a.format],dest):
            failed.append(r["File"]); continue
        done.append({**{k:r.get(k,"") for k in FIELDS},"path":dest})
    m=update_manifest(a.out,done)
    print(f"\n{len(done)} image(s) disponibles, {len(failed)} echec(s). Manifest : {m}")
    if failed:
        print("echecs :"," ".join(failed))


if __name__=="__main__":
    main()