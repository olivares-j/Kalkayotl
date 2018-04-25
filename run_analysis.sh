#/bin/bash
python analysis_p2d.py Uniform 0 500 &
sleep 1
python analysis_p2d.py Uniform 0 1000 &
sleep 1
python analysis_p2d.py Gaussian 250 100 &
sleep 1
python analysis_p2d.py Gaussian 250 500 &
sleep 1
python analysis_p2d.py Gaussian 250 1000 &
sleep 1
python analysis_p2d.py Cauchy 250 100 &
sleep 1
python analysis_p2d.py Cauchy 250 500 &
sleep 1
python analysis_p2d.py Cauchy 250 1000 &
sleep 1
python analysis_p2d.py EDSD 0 100 &
sleep 1
python analysis_p2d.py EDSD 0 500 &
sleep 1
python analysis_p2d.py EDSD 0 1000 &
sleep 1
python analysis_p2d.py EDSD 0 1350 &
sleep 1
python analysis_p2d.py EDSD 0 1500 &