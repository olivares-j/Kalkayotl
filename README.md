# Kalkayotl
Kalkayotl is a Bayesian inference code designed to obtain posterior estimate of cluster parameters, this is location and scale, and distance to the cluster stars.

## Installation
================

1. Get the code from https://github.com/olivares-j/Kalkayotl, you can either clone or download. Remember to use the master branch.


2. I strongly recommend to create an independent environment (see for example: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html, this will avoid mixing python packages). This new environment must have python 3.6 or higher.
You can do this by Anaconda with the following command:

```
conda create -n myenv python=3.6
```
with `myenv` the name of your choice.

3. Move into the new environment

```conda activate myenv```

with `myenv` the name of the environment.

4. Install the following packages:

```
conda install -c conda-forge pymc3
conda install -c conda-forge matplotlib
conda install -c conda-forge dynesty
conda install -c conda-forge arviz
```

5. Navigate to the Kalkayotl folder and install the package:

```
pip install dist/Kalkayotl-0.3.0.tar.gz
```

6. Test the installation by running

```
python example.py
```

It will compute cluster and star distances using the Ruprecht_147.csv data from the Data folder. You must get a Test folder with the outputs.

Whenever you run Kalkayotl, remember to move into its environment (step 3).

