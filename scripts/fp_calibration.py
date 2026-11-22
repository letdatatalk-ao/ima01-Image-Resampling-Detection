"""Calibration sous H0 sur les images ORIGINALES (pas de reechantillonnage) :
proportion d'images avec au moins une detection en fonction du seuil eps.
Le cadre a contrario garantit E[#fausses detections par image] <= eps, donc cette proportion doit rester <= eps.
Usage : python scripts/fp_calibration.py [--dir results/final]
"""
import argparse, glob, os
import numpy as np
import pandas as pd
from scipy.stats import beta

EPS=[1.0,0.1,0.01,0.001]


def ci(k,n):
    return (100*beta.ppf(.025,k,n-k+1) if k else 0.0, 100*beta.ppf(.975,k+1,n-k) if k<n else 100.0)


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir",default="results/final"); a=ap.parse_args(); rows=[]
    for p in sorted(glob.glob(os.path.join(a.dir,"detection_*.csv"))):
        r=pd.read_csv(p); o=r[(r["label"]=="original")&(r["detected"].astype(str)!="ERROR")]
        if len(o)==0: continue
        name=os.path.basename(p)[len("detection_"):-4]
        for mode,col in [("avec c.v.","log10_nfa"),("sans c.v.","log10_nfa_nocv")]:
            if col not in o: continue
            for e in EPS:
                k=int((o[col]<np.log10(e)).sum()); n=len(o); lo,hi=ci(k,n)
                rows.append({"resultats":name,"mode":mode,"eps":e,"FP images":f"{k}/{n}","FP %":round(100*k/n,1),
                             "IC95 %":f"[{lo:.1f} ; {hi:.1f}]","borne eps %":100*e})
    df=pd.DataFrame(rows); print(df.to_string(index=False))
    out=os.path.join(a.dir,"fp_calibration.csv"); df.to_csv(out,index=False); print("\n->",out)


if __name__=="__main__":
    main()