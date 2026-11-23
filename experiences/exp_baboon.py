"""EXP 4 - Images de reference des auteurs (rapport Section 6.2).
baboon.png (512) et baboon_666.png (512 -> 666) : PNG, rank transform.
pashmina.jpg (800) et pashmina_720.jpg (800 -> 720) : JPEG, residu TV et suppression des pics kN/8.
Attendu : rien sur les originaux, detection a d = +-M mod N sur les images reechantillonnees.
Usage : python experiences/exp_baboon.py [dossier des images, defaut data/ird_img]"""
import sys,os
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.image_io import load_gray
from src.detector import detect

FOLDER=sys.argv[1] if len(sys.argv)>1 else "data/ird_img"
CASES=[("baboon.png",None),("baboon_666.png",512),("pashmina.jpg",None),("pashmina_720.jpg",800)]   # (fichier, M d'origine)

for name,M in CASES:
    img=load_gray(os.path.join(FOLDER,name)); N=img.shape[0]
    is_jpeg=name.endswith(".jpg")
    o=detect(img,residual="tv" if is_jpeg else "rt",jpeg=is_jpeg)
    expected=sorted({M%N,(-M)%N}) if M else []
    print(f"{name:18s} {img.shape} detecte={o['detected']!s:5s} d_V={[int(x) for x in o['d_v']]} "
          f"d_H={[int(x) for x in o['d_h']]} attendu={expected} log10 NFA={o['log10_nfa']:.1f}")
