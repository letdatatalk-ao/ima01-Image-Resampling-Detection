"""Ablation de l'extraction du residu (Table IX du papier) : precision (%) a 1 % de faux positifs,
noyau bicubique, pour chaque residu disponible dans results/final/detection_<format>_<residu>.csv.

Prealable (exemple PNG) :
    python scripts/run_detection.py --format png --kernel cubic --residual none --workers 4
    python scripts/run_detection.py --format png --kernel cubic --residual rt   --workers 4
    python scripts/run_detection.py --format png --kernel cubic --residual tv   --workers 4
Usage : python scripts/ablation.py [--dir results/final] [--kernel cubic]
"""
import argparse, glob, os
import numpy as np
import pandas as pd

QS=[0.6,0.7,0.8,0.9,1.1,1.2,1.3,1.4,1.5,1.6]


def threshold_1pct(s):
    s=np.sort(np.asarray(s,dtype=float)); return s[int(np.floor(0.01*len(s)))]


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir",default="results/final"); ap.add_argument("--kernel",default="cubic")
    a=ap.parse_args(); rows=[]
    for p in sorted(glob.glob(os.path.join(a.dir,"detection_*.csv"))):
        fmt,res=os.path.basename(p)[len("detection_"):-4].split("_",1)
        r=pd.read_csv(p); r=r[r["detected"].astype(str)!="ERROR"]
        o=r[r["label"]=="original"]; s=r[(r["label"]=="resampled")&(r["kernel"]==a.kernel)]
        if len(o)==0 or len(s)==0 or "log10_nfa" not in r: continue
        t=threshold_1pct(o["log10_nfa"])
        acc={q:100*(s[s["q"]==q]["log10_nfa"]<t).mean() for q in QS if (s["q"]==q).any()}
        rows.append({"format":fmt,"residual":res,"threshold_log10":round(t,2),**{str(q):round(v,1) for q,v in acc.items()},
                     "mean":round(np.mean(list(acc.values())),1),"n_orig":len(o)})
    if not rows:
        print("aucun resultat dans",a.dir); return
    df=pd.DataFrame(rows).sort_values(["format","residual"])
    print(f"Precision (%) a 1 % de faux positifs, noyau {a.kernel}, avec validation croisee\n")
    print(df.to_string(index=False))
    out=os.path.join(a.dir,"ablation.csv"); df.to_csv(out,index=False); print("\n->",out)


if __name__=="__main__":
    main()