# Kalkayotl
Kalkayotl is a Bayesian inference code designed to obtain samples of the joint posterior distribution of cluster parameters (so far only location and scale) and distances to the cluster stars.

**Note that there is not yet a manual of the code since the 3D and 6D versions are currently under development. In the meantime, please read carefully the example.py file and its comments as well as the Tutorial.ipynb file. These files explain the use of the code in its current first release. Specific details of the prior families and the undertaken assumptions are given in the associated paper.**



## Installation

1. Get the code from https://github.com/olivares-j/Kalkayotl, you can fork, clone, or download. Remember to use the master branch which contains the already tested 1D version.


2. I strongly recommend creating an independent conda environment (see for example: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html, this will avoid mixing python packages). This new environment must have python 3.6.
You can do this by Anaconda with the following command:

**Linux users**
```
conda create -n kalkayotl -c conda-forge python=3.6.10 pymc3=3.7 matplotlib=3.1.3 dynesty=1.0.0 arviz=0.5.1
```
**Mac OS users**
```
conda create -n kalkayotl -c conda-forge clang=4.0.1 python=3.6.10 pymc3=3.7 matplotlib=3.1.3 dynesty=1.0.0 arviz=0.5.1
```
Note: PyMC3 may have problems to run in old operative systems (e.g. MAC OS < 10.14)

The option `kalkayotl` is the name of the environment and you can choose another name.

Newer versions of these libraries may cause conflicts amongst them, stick to these.

3. Move into the newly created `kalkayotl` environment

```conda activate kalkayotl```


4. Test pymc3 installation:

Open a python console and type:
```import pymc3```

If an error appears follow the [PyMC3](https://docs.pymc.io/) installation instructions.

If asked for, then also install the mkl-service by typing within the Kalkayotl environment:
`conda install mkl-service`

Note: if you want to run the Tutorial notebook you also need to install `jupyterlab` in the same environment:
```
conda install -c conda-forge jupyterlab
```



5. Install Kalkayotl:

Once you have successfully installed PyMC3, fork, clone, or download the Kalkayotl repository, move into it and type:

```
pip install dist/Kalkayotl-1.0.tar.gz
```

Test the installation by running

```
python example.py
```

It will compute cluster and star distances using the Ruprecht_147.csv data from the Data folder. You must get an Example folder with the outputs (chains, statistics, and plots).

Whenever you run Kalkayotl, remember to move into its environment (step 3).

## Running the code

The easiest way to run the code on your own data sets is to copy the ``example.py`` file and modify it according to your needs. Instructions are given within it. Please read it carefully, especially the comments.

Before running Kalkayotl:

1. Verify that your input file contains the typical Gaia columns. You can compare it with the input file of the example. 
2. Remove possible duplicated sources and/or duplicated identifiers from the input file.
3. Execute the file:```python example.py```

For more information open the Tutorial notebook by typing:
`jupyter notebook Tutorial.ipynb`

## Troubleshooting

The most common errors that you may face while running Kalkayotl are:

1. ``RuntimeError: Chain failed.``
 This error is caused generally by a zero derivative in a random variable (RV). In most cases, it is solved by running the code again, which will initialize the chain in another point of parameter space. Remember that you must manually remove the files (chain-?.csv) to avoid reusing the positions of those failed chains.

2. Low effective sample size and/or divergences.
 The first is caused by poor sampling while divergences are related to numerical issues. Few effective samples will result in low parameter precision. Several divergences indicate that the posterior is hard to sample, usually because the data is not informative enough to constrain the parameters of a complex model. However, a few divergences are generally not an issue, but still, take a look at the chains.

 Possible solutions:
 * Increase the number of tuning iterations. 
 * Increase the ``target_accept`` parameter of the sampler: from 0.8 to 0.9 or 0.95. 
 * Constrain the model by adding prior information in the hyper-parameters (e.g. set hyper_beta to 10 or 20 pc).
 * Testing the two types of parameterization: "central" and "non-central". The former works better for constraining data sets (i.e. populous and nearby clusters at less than 500 pc).
 * Fix some parameters, like gamma= 5 in the EFF, which will produce a Plummer profile.

 Advice: Whenever possible use simpler models.

 As noted in the article, the Gaussian Mixture Model is problematic due to its complexity. If you absolutely need it, I strongly recommend computing statistics with only one chain (remove or rename chain-1.csv, and recompute the statistics). Due to the lack of identifiability, the Gaussian components can be interchanged. For example component A and B are first and second in one chain and second and first in other then chain since statistics are computed with the mixed chains the results are no longer correct.
 
**If you face problems during the installation or while running the code, please send an issue instead of an e-mail. Your question/issue may help other users.**

Finally, if you have comments, improvements, or suggestions please let me know.

