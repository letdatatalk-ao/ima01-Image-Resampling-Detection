@echo off
REM Relance complete avec le controle des fausses alarmes corrige (un patch par paire conjuguee).
REM A lancer depuis code_ima, venv active :  run_all_corrige.bat
REM Duree estimee avec 4 workers : PNG ~35 min, JPEG95 ~2 h, JPEG90 ~2 h, le reste ~30 min.
REM Les resultats du papier (tous les patches) restent dans results\final ; les nouveaux vont dans results\corrige.
set W=4

python -m pytest -q || goto :fin
python experiences\exp_h0_bruit.py > results\h0_bruit.txt
type results\h0_bruit.txt

python scripts\run_detection.py --format png --workers %W%
python scripts\run_detection.py --format png --kernel cubic --residual none --workers %W%
python scripts\run_detection.py --format png --kernel cubic --residual tv --workers %W%
python scripts\run_detection.py --data data\resizers --format png --workers %W% --out results\corrige\detection_png_rt_pil.csv
python scripts\run_detection.py --format jpeg95 --workers %W%
python scripts\run_detection.py --format jpeg90 --workers %W%

python scripts\table4.py --dir results\corrige > results\corrige\table4.txt
python scripts\ablation.py --dir results\corrige > results\corrige\ablation.txt
python scripts\fp_calibration.py --dir results\corrige > results\corrige\fp_calibration.txt
python video_data.py

REM Optionnel (hypothese de l'ecart JPEG) : distances exclues kN/16 comme le code des auteurs
python scripts\run_detection.py --format jpeg90 --kernel cubic --jpeg-block 16 --workers %W%

:fin
echo Termine.
