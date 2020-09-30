# Kalkayotl
Kalkayotl is a Bayesian inference code designed to obtain posterior estimate of cluster parameters, this is location and scale, and distance to the cluster stars.

## Installation

1. Get the code from https://github.com/olivares-j/Kalkayotl, you can either clone or download. Remember to use the master branch.


2. I strongly recommend to create an independent environment (see for example: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html, this will avoid mixing python packages). This new environment must have python 3.6.
You can do this by Anaconda with the following command:

** Linux users **
```
conda create -n kalkayotl -c conda-forge python=3.6.10 pymc3=3.7 matplotlib=3.1.3 dynesty=1.0.0 arviz=0.5.1
```
** Mac OS users **
```
conda create -n kalkayotl -c conda-forge clang=4.0.1 python=3.6.10 pymc3=3.7 matplotlib=3.1.3 dynesty=1.0.0 arviz=0.5.1
```
Note: PyMC3 may have problems to run in old operative systems (e.g. MAC OS < 10.14)

with `kalkayotl` the name of your choice.

Note that newer versions of these libraries may cause conflicts amongst them. So, please stick to these ones.

3. Move into the new environment

```conda activate kalkayotl```

with `kalkayotl` the name of the environment that we just created.

4. Test pymc3 installation:

Open a python console and type:
```import pymc3```

If an error appears follow the instructions of [PyMC3](https://docs.pymc.io/)

If asked for also install mkl-service by typing within the Kalkayotl environement:
`conda install mkl-service`

Note: if you want to run the nootebook you also need to install `jupyterlab`:
```
conda install -c conda-forge jupyterlab
```



5. Install Kalkayotl:

Once you have cloned or forked the Kalkayotl repository move into its directory and type:

```
pip install dist/Kalkayotl-1.0.tar.gz
```

Test the installation by running

```
python example.py
```

It will compute cluster and star distances using the Ruprecht_147.csv data from the Data folder. You must get an Example folder with the outputs (chains, statistics and plots).

Whenever you run Kalkayotl, remember to move into its environment (step 3).

## Running the code

The easiest way to run the code on your own data sets is to copy the ``example.py`` file and modify it according to your needs. Instructions are given within it. Please read it carefully.

Before running Kalkayotl:

1. Verify that your input file contains the typical Gaia columns. You can compare with the input file of the example. 
2. Remove possible duplicated sources and/or duplicated identifiers from the input file.
3. Execute the file. E.g. ```python example.py```

For additional details see the Tutorial notebook:
`jupyter notebook Tutorial.ipynb`

## Troubleshooting


The most common errors that you may face while running Kalkayotl are:

1. ``RuntimeError: Chain failed.``
 This error is caused generally by a zero derivative in a random variable (RV). In most cases it is solved by running the code again, which will initialize the chain in another point of parameter space. Remember that you must manually remove the files (chain-?.csv) in order to avoid reusing the positions of those failed chains.

2. Low effective sample size and/or divergences.
 The first is caused by a poor sampling while divergences are related to numerical issues. While very few effective samples means that the parameter precision is lower, the divergences indicate that the posterior is hard to sample. A few divergences are generally not an issue, but still look the chains.

 Possible solutions:
 * Increase the number tuning iterations. 
 * Increase the ``target_accept`` parameter of the sampler: from 0.8 to 0.9 or 0.95. 
 * Constrain the model by adding adding prior information in the hyper-parameters (e.g. set hyper_beta to 10 or 20 pc).
 * Testing the two types of parameterization: "central" and "non-central". The former works better for nearby clusters (<500 pc).
 * Fixing some parameter like gamma= 5 in the EFF, which will produce a Plummer profile.

 Advice: Whenever possible use simpler models.

 If you absolutely need the GMM prior I strongly recommend to compute statistics with only one chain (i.e. remove or rename chain-1.csv, and recompute the statistics). Due to lack of identifiability the Gaussians components can be interchanged. For example component A and B are first and second in one chain and second and first in other chain, since statistics are computed with the mixed chains the results are no longer correct.

 If nothing of the above solves your problem create a GitHub issue explaining the problem and the error message.

