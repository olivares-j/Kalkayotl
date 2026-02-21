# Kalkayotl
<a href="https://ascl.net/2011.003"><img src="https://img.shields.io/badge/ascl-2011.003-blue.svg?colorB=262255" alt="ascl:2011.003" /></a>

Kalkayotl is a Bayesian inference code designed to obtain samples of the joint posterior distribution of cluster parameters and individual positions and velocities of the cluster stars.


**Note that there is not yet a manual of the code since the 3D and 6D versions are currently under development. In the meantime, please read carefully the example.py file and its comments. These files explain the use of the code in its current release. Specific details of the prior families and the undertaken assumptions are given in the associated papers [paperI](
https://www.aanda.org/articles/aa/pdf/2020/12/aa37846-20.pdf) and [paperII](https://www.aanda.org/articles/aa/pdf/2025/01/aa48362-23.pdf).**

## Updates
- This is the 2.1 version which now allows 6D modelling of stellar systems.
- The parallax spatial correlation of Lindegren et al. 2020 (Gaia eDR3) is now included as the default one in version 1.1.



## Installation

1. Get the code from https://github.com/olivares-j/Kalkayotl, you can fork, clone, or download. For the 1D version use the branch labelled v1.0. For the 3D and 6D versions use the v2.0 branch.


2. I strongly recommend creating an independent conda environment (see for example: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html, this will avoid mixing python packages). This new environment must have python 3.11.
You can do this by Anaconda with the following command:

```
conda create -n kalkayotl -c conda-forge pymc arviz numpy scipy astropy pandas h5py dill
```
The previous line will provide the basic installation. Depending on your machine and demands, accelerators like JAX will need to be installed in CPU or GPU version. 

3. Move into the newly created `kalkayotl` environment

```conda activate kalkayotl```

For CPUs:
```
conda install -c conda-forge jaxlib jax
```

For Nvidia GPUs: 
```
conda install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia
```

Finally, for the use of more efficient sampling methods install blackjax and numpyro:
```

conda install -c conda-forge blackjax numpyro
```

4. Test pymc installation:

Open a python console and type:
```import pymc```

It should be loaded silently. However, if an error occurs follow the [PyMC](https://docs.pymc.io/) installation instructions.

5. Install Kalkayotl:

Once you have successfully installed PyMC move to the Kalkayotl directory (the one you forked, cloned, or downloaded in step 1) and type:

```
pip install dist/kalkayotl-2.1.0-py3-none-any.whl
```

Test the installation by running

```
python example.py
```

It will infer the source-level and cluster-level parameters of the Beta Pictoris stellar association using the provided data. You must get the outputs (chains, statistics, and plots) within the same Example folder. If you have no errors then you are ready to move to the next section. If errors appear, identify if they are related to Kalkayotl, PyMC, or the dependencies. If they are related to PyMC or the dependencies follow the specific instructions in their web pages. If it is related to the installation of Kalkayotl, then submit an issue explaining the error. 

**Mac OS users**

In the past, Mac OS users experienced issues while installing pymc due to different gcc compilers. So I do recommend to first install pymc in the new conda environment test it and proceed to install the rest of the packages. 

**NOTE on PyMC error**
If you get the following error:
```
IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices
```
at line 154 of the (your_installation_path_for_conda_environement_kalkayotl)/pymc/distributions/transforms.py file, then it means the PyMC team has not yet solved this issue. To fix it, open the mentioned file and do the following modifications.

1. After the line ```from pytensor.tensor import TensorVariable``` (around line 22 in version 5.27) add the following line:
```
from pytensor.tensor.basic import tensor_copy
``` 

2. Identify the class "CholeskyCovPacked" (around line 139 in version 5.27) and do the following modifications:

3. Replace line ```self.diag_idxs = pt.arange(1, n + 1).cumsum() - 1``` by ```self.diag_idxs = pt.arange(1, n.value + 1).cumsum() - 1``` (This simply evaluates n so that arange can work).

4. Within the "backward" function, add the following line as the first line of the function ```x = tensor_copy(value)``` (This simply copies the variable "value" into variable "x").

5. Within the same "backward" function replace line ```return pt.set_subtensor(value[..., self.diag_idxs], pt.exp(value[..., self.diag_idxs]))``` by the following two lines:
    ```
    x = pt.set_subtensor(x[...,self.diag_idxs], pt.exp(x[...,self.diag_idxs]))
    return x
    ```
    (This do the same original operation on the copied variable "x" instead of the original one "value" and returns it).

6. Within the "forward" function, add the following line as the first line of the function ```x = tensor_copy(value)``` (This simply copies the variable "value" into variable "x").

7. Within the same "forward" function replace line ```return pt.set_subtensor(value[..., self.diag_idxs], pt.log(value[..., self.diag_idxs]))``` by the following two lines:
    ```
    x = pt.set_subtensor(x[..., self.diag_idxs], pt.log(x[..., self.diag_idxs]))
        return x
    ```
    (This do the same original operation on the copied variable "x" instead of the original one "value" and returns it).

8. Save the file and test the example.py code again. This should have solved the problem for PyMC version up to 5.27. If you continue to have problems, please, contact me.


## Running the code

Whenever you run Kalkayotl, remember to move into its environment by typing ``conda activate kalkayotl`` (or the name that you use at the installation).

The easiest way to run the code on your own data sets is to copy the ``example.py`` file and modify it according to your needs. Instructions are given within it. Please read it carefully, especially the comments.

Before running Kalkayotl:

1. Verify that your input file contains the typical Gaia columns. You can compare it with the input file of the example. 
2. Remove possible duplicated sources and/or duplicated identifiers from the input file.
3. Execute the file:```python example.py```


## Troubleshooting

The most common errors that you may face while running Kalkayotl are:

1. ``RuntimeError: Chain failed.``
 This error is caused generally by a zero derivative in a random variable (RV). In most cases, it is solved by running the code again, which will initialize the chain in another point of parameter space. Remember that you must manually remove the files (chain-?.csv) to avoid reusing the positions of those failed chains.

2. Low effective sample size and/or divergences.
 The first is caused by poor sampling while divergences are related to numerical issues. Few effective samples will result in low parameter precision. Several divergences indicate that the posterior is hard to sample, usually because the data is not informative enough to constrain the parameters of a complex model. However, a few divergences are generally not an issue, but still, take a look at the chains.

 Possible solutions:
 * Increase the number of tuning iterations. 
 * Increase the ``target_accept`` parameter of the sampler: from 0.65 to 0.9 or 0.95. 
 * Constrain the model by adding prior information in the hyper-parameters (e.g. set hyper_beta to 10 or 20 pc).
 * Testing the two types of parameterization: "central" and "non-central". The former works better for constraining data sets (i.e. populous and nearby clusters at less than 500 pc).
 * Fix some parameters, like gamma= 5 in the EFF, which will produce a Plummer profile.

 Advice: Whenever possible use simpler models.

 As noted in the article, the Gaussian Mixture Model is problematic due to its complexity. If you absolutely need it, I strongly recommend computing statistics with only one chain . Due to the lack of identifiability, the Gaussian components can be interchanged. For example component A and B are first and second in one chain and second and first in the other chain. Given that statistics are computed with the mixed chains the results are no longer correct.
 
**If you face problems during the installation or while running the code, please send an issue instead of an e-mail. Your question/issue may help other users.**

Finally, if you have comments, improvements, or suggestions, please let me know. .... and do not forget to cite the v1.0 [paper](
http://arxiv.org/abs/2010.00272) and the v2.0 [paper](https://arxiv.org/abs/2411.16012) if you use the code ;).

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

@ARTICLE{2025A&A...693A..12O,
       author = {{Olivares}, J. and {Bouy}, H. and {Dorn-Wallenstein}, T.~Z. and {Berihuete}, A.},
        title = "{Kalkayotl 2.0: Bayesian phase-space modelling of star-forming regions, stellar associations, and open clusters}",
      journal = {\aap},
     keywords = {methods: statistical, stars: kinematics and dynamics, open clusters and associations: general, open clusters and associations: individual: {\ensuremath{\beta}} Pictoris, open clusters and associations: individual: Hyades, open clusters and associations: individual: Praesepe, Astrophysics - Astrophysics of Galaxies, Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Solar and Stellar Astrophysics},
         year = 2025,
        month = jan,
       volume = {693},
          eid = {A12},
        pages = {A12},
          doi = {10.1051/0004-6361/202348362},
archivePrefix = {arXiv},
       eprint = {2411.16012},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025A&A...693A..12O},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}


```
