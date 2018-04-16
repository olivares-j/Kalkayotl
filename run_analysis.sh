#/bin/bash
python analysis_p2d.py Uniform 5 &
python analysis_p2d.py Gaussian 1 &
python analysis_p2d.py Gaussian 5 &
python analysis_p2d.py Gaussian 10 &
python analysis_p2d.py Cauchy 1 &
python analysis_p2d.py Cauchy 5 &
python analysis_p2d.py Cauchy 10 &
python analysis_p2d.py EDBJ2015 1 &
python analysis_p2d.py EDBJ2015 5 &
python analysis_p2d.py EDBJ2015 10 &