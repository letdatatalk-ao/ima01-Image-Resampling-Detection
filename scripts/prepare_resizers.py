"""Images reechantillonnees par une VRAIE bibliotheque (Pillow), comme les Tables V-VI du papier.
Notre modele (resample_2d_fast) n'a pas d'anti-repliement ; Pillow, lui, elargit le filtre quand il
reduit l'image (anti-aliasing), ce qui attenue la trace en sous-echantillonnage.

Meme protocole que prepare_raise.py : TIFF -> luminance -> x1/2 -> carre central S_in -> Pillow (S_in -> S_out)
-> centre 600x600 -> PNG.  Les originales sont reprises telles quelles (q = 1).
Sortie : data/resizers/{original,resampled/pil_<noyau>/q<q>}/png/<id>.png + index.csv (lisible par run_detection.py)

    python scripts/prepare_resizers.py --n 200 --kernels cubic --workers 4
    python scripts/run_detection.py --data data/resizers --format png --workers 4 --out results/final/detection_png_rt_pil.csv
"""
import argparse, csv, os, sys
from multiprocessing import Pool
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.image_io import load_gray, save_gray_png
from src.dataset import FACTORS, OUT_SIZE, downsample2, center_crop, crop_size_for, expected_distances

PIL_FILTERS={"nearest":Image.Resampling.NEAREST,"linear":Image.Resampling.BILINEAR,
             "cubic":Image.Resampling.BICUBIC,"lanczos":Image.Resampling.LANCZOS}
COLS=["id","device","label","kernel","q","format","path","size","d1","d2","source"]


def resize_pil(img,q,kernel,out=OUT_SIZE):
    """Carre central S_in -> Pillow -> S_out (rapport exactement q) -> centre out x out."""
    s_in,s_out=crop_size_for(q,out)
    region=Image.fromarray(center_crop(img,s_in).astype(np.float32),mode="F")
    return center_crop(np.asarray(region.resize((s_out,s_out),resample=PIL_FILTERS[kernel]),dtype=np.float64),out)


def process(job):
    row,out,kernels=job; iid,src=row["File"],row["path"]; rows=[]
    try:
        half=downsample2(load_gray(src))
        p=os.path.join(out,"original","png",f"{iid}.png")
        if not os.path.exists(p): save_gray_png(center_crop(half,OUT_SIZE),p)
        rows.append({"id":iid,"device":row.get("Device",""),"label":"original","kernel":"none","q":1.0,"format":"png",
                     "path":os.path.relpath(p,out),"size":OUT_SIZE,"d1":"","d2":"","source":src})
        for k in kernels:
            for q in FACTORS:
                p=os.path.join(out,"resampled",f"pil_{k}",f"q{q:.1f}","png",f"{iid}.png")
                if not os.path.exists(p): save_gray_png(resize_pil(half,q,k),p)
                d1,d2=expected_distances(q)
                rows.append({"id":iid,"device":row.get("Device",""),"label":"resampled","kernel":f"pil_{k}","q":q,"format":"png",
                             "path":os.path.relpath(p,out),"size":OUT_SIZE,"d1":round(d1,2),"d2":round(d2,2),"source":src})
        return iid,rows,"ok"
    except Exception as e:
        return iid,[],f"ECHEC : {e}"


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest",default="data/raise/tiff/manifest.csv"); ap.add_argument("--out",default="data/resizers")
    ap.add_argument("--n",type=int,default=None); ap.add_argument("--kernels",nargs="+",default=["cubic"],choices=list(PIL_FILTERS))
    ap.add_argument("--workers",type=int,default=1); a=ap.parse_args()
    with open(a.manifest,newline="",encoding="utf-8") as f:
        photos=[r for r in csv.DictReader(f) if os.path.exists(r["path"])]
    photos=photos[:a.n] if a.n else photos
    jobs=[(r,a.out,a.kernels) for r in photos]; allrows=[]
    it=Pool(a.workers).imap(process,jobs) if a.workers>1 else map(process,jobs)
    for i,(iid,rows,msg) in enumerate(it,1):
        print(f"[{i}/{len(photos)}] {iid} : {msg}",flush=True); allrows+=rows
    os.makedirs(a.out,exist_ok=True)
    with open(os.path.join(a.out,"index.csv"),"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=COLS); w.writeheader(); w.writerows(allrows)
    print("index ->",os.path.join(a.out,"index.csv"))


if __name__=="__main__":
    main()