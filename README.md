# Kalkayotl
Kalkayotl is a Bayesian inference code designed to obtain posterior estimate of cluster parameters, this is location and scale, and distance to the cluster stars.

===== Installation ======

1. Get the code from https://github.com/olivares-j/Kalkayotl, you can either clone or download. Remember to use the pymc3 branch.


2. I strongly recommend to create an independent environment (this will avoid mixing python packages). This new environment must have python 3.6 or higher.
You can do this by Anaconda with the following command:

```
conda create -n myenv python=3.6
```
with `myenv` the name of your choice, like kalkayotl ;).

3. Move into the new environment (i.e. `conda activate myenv`), and navigate to the Kalkayotl folder.

4. Install the package with

```
pip install dist/Kalkayotl-0.3.0.tar.gz
```

5. Test the installation by running

```
python example.py
```
It will compute cluster and star distances using the Ruprecht_147.csv data from the Data folder. You must get a Test folder with the outputs.

