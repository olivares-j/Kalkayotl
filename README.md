# Kalkayotl
Kalkayotl is a Bayesian inference code designed to obtain posterior estimate of cluster parameters, this is location and scale, and distance to the cluster stars.

## Installation

1. Get the code from https://github.com/olivares-j/Kalkayotl, you can either clone or download. Remember to use the master branch.


2. I strongly recommend to create an independent environment (see for example: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html, this will avoid mixing python packages). This new environment must have python 3.6 or higher.
You can do this by Anaconda with the following command:

```
conda create -n myenv python=3.6.10
```
with `myenv` the name of your choice.

3. Move into the new environment

```conda activate myenv```

with `myenv` the name of the environment.

4. Install the following packages:

```
conda install -c conda-forge pymc3=3.7
conda install -c conda-forge matplotlib=3.1.3
conda install -c conda-forge dynesty=1.0.0
conda install -c conda-forge arviz=0.5.1
```

5. Navigate to the Kalkayotl folder and install the latest package:

```
pip install dist/Kalkayotl-0.9.0.tar.gz
```

6. Test the installation by running

```
python example.py
```

It will compute cluster and star distances using the Ruprecht_147.csv data from the Data folder. You must get a Test folder with the outputs.

Whenever you run Kalkayotl, remember to move into its environment (step 3).

## Running the code

The easiest way to run the code on your own data sets is to copy the ``example.py`` file and modify it according to your needs. Instructions are given within it, read it carefully.

Before running Kalkayotl:

1. Verify that your input file contains the typical Gaia columns. You can compare with the input file of the example. 
2. Remove possible duplicated sources and/or duplicated identifiers from the input file.
3. Execute the file. E.g. ```python example.py```

## Troubleshooting


The most common errors that you may face while running Kalkayotl are:

1. ``RuntimeError: Chain failed.``
 This error is caused generally by a zero derivative in a random variable (RV). In most cases it is solved by running the code again, which will initialize the chain in another point of parameter space. Remember that you must manually remove the files (chain-?.csv) in order to avoid reusing the positions of those failed chains.

2. Low effective sampler size and/or divergences.
 The first is caused by a poor sampling while divergences are related to numerical issues. In both cases try to run with more tuning iterations. Another option is to increase the ``target_accept`` parameter of the sampler. 

 Finally, if you still have convergence problems try to reparametrize the model by either fixing some parameter (e.g. gamma:5 in the EFF will produce a Plummer profile) and/or constrain the prior by changing the hyper-parameters (set hyper_beta to 10.0 instead of 100).

 Advice: Whenever possible use simpler models.

 If nothing of the above solves your problem create a GitHub issue explaining the problem and the error message.

