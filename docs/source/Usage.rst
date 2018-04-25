.. _Usage:

Usage
--------------

Once installed, to run the code you need to first import the p2d function parallax2distance

.. code-block:: Python

	from p2d import parallax2distance

then, you will have to initialise the module with the information concerning the prior, the number of MCMC iteration. For example

.. code-block:: Python
	p2d = parallax2distance(N_iter=1000,prior="Uniform",prior_loc=0,prior_scale=500)

Finally, to run you will need to call the run function with your datum, as

.. code-block:: Python
	MAP,Mean,SD,CI,int_time,sample = p2d.run(plx,u_plx)

where plx and u_plx are the parallax and its uncertainty.

The the parallax2distance.run function returns the MAximum-A-Posteriori, the mean, standard deviation and the confidence interval (95%) of the MCMC samples.
In case you need extra information, it also returns the integrated autocorrelation time of the MCMC and its entire sample (burnin removed).