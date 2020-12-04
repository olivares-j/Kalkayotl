# Kalkayotl
<a href="https://ascl.net/2011.003"><img src="https://img.shields.io/badge/ascl-2011.003-blue.svg?colorB=262255" alt="ascl:2011.003" /></a>

Kalkayotl is a Bayesian inference code designed to obtain samples of the joint posterior distribution of cluster parameters (so far only location and scale) and distances to the cluster stars.

**Note that there is not yet a manual of the code since the 3D and 6D versions are currently under development. In the meantime, please read carefully the example.py file and its comments as well as the Tutorial.ipynb file. These files explain the use of the code in its current first release. Specific details of the prior families and the undertaken assumptions are given in the associated [paper](
https://www.aanda.org/articles/aa/pdf/2020/12/aa37846-20.pdf).**

## Updates

- The parallax spatial correlation of Lindegren et al. 2020 (Gaia eDR3) is now included as the default one in version 1.1.



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

It should be loaded silently. However, if an error occurs follow the [PyMC3](https://docs.pymc.io/) installation instructions.

If you have a warning about the mkl-service library, install it by typing:
`conda install mkl-service`
within the Kalkayotl environment.

Note: if you want to run the Tutorial notebook you also need to install `jupyterlab` in the same environment:
```
conda install -c conda-forge jupyterlab
```

5. Install Kalkayotl:

Once you have successfully installed PyMC3 move to the Kalkayotl directory (the one you forked, cloned, or downloaded in step 1) and type:

```
pip install dist/Kalkayotl-1.1.tar.gz
```

Test the installation by running

```
python example.py
```

It will compute cluster and star distances using the Ruprecht_147.csv data from the Example folder. You must get the outputs (chains, statistics, and plots) within the same Example folder. If you have no errors then you are ready to move to the next section. If errors appear, identify if they are related to Kalkayotl, PyMC3, or the dependencies. If they are related to PyMC3 or the dependencies follow the specific instructions in their web pages. If it is related to the installation of Kalkayotl, then submit an issue explaining the error. 



## Running the code

Whenever you run Kalkayotl, remember to move into its environment by typing ``conda activate kalkayotl`` (or the name that you use in step 3 of the installation).

The easiest way to run the code on your own data sets is to copy the ``example.py`` file and modify it according to your needs. Instructions are given within it. Please read it carefully, especially the comments.

Before running Kalkayotl:

1. Verify that your input file contains the typical Gaia columns. You can compare it with the input file of the example. 
2. Remove possible duplicated sources and/or duplicated identifiers from the input file.
3. Execute the file:```python example.py```

For more information open the Tutorial notebook by typing:
`jupyter notebook Tutorial.ipynb`
always within the Kalkayotl environement created in the installation steps.

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

Finally, if you have comments, improvements, or suggestions, please let me know. .... and do not forget to cite the [paper](
http://arxiv.org/abs/2010.00272) if you use the code ;).

### Citation
```
@ARTICLE{2020A&A...644A...7O,
       author = {{Olivares}, J. and {Sarro}, L.~M. and {Bouy}, H. and {Miret-Roig}, N. and {Casamiquela}, L. and {Galli}, P.~A.~B. and {Berihuete}, A. and {Tarricq}, Y.},
        title = "{Kalkayotl: A cluster distance inference code}",
      journal = {\aap},
     keywords = {methods: statistical, parallaxes, open clusters and associations: general, stars: distances, virtual observatory tools, Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Astrophysics of Galaxies, Astrophysics - Solar and Stellar Astrophysics},
         year = 2020,
        month = dec,
       volume = {644},
          eid = {A7},
        pages = {A7},
          doi = {10.1051/0004-6361/202037846},
archivePrefix = {arXiv},
       eprint = {2010.00272},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020A&A...644A...7O},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

