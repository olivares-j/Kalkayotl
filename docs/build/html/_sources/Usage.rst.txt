.. _Usage:

Usage
--------------

Once installed, to run the code you need to first import the parallax2distance function from the p2d file

.. code-block:: python

	from p2d import parallax2distance

then, you will have to initialise the module with the information concerning the prior, the the number of MCMC iteration, and the burning fraction to discard (by default the initial 20% of the chain). For example:

.. code-block:: python

	p2d = parallax2distance(N_iter=1000,prior="Uniform",prior_loc=0,prior_scale=500)

Notice that the units for the prior location and scale are in parsecs.

Finally, to obtain the distance you need to call the run function with your datum:

.. code-block:: Python

	MAP,Mean,SD,CI,int_time,sample = p2d.run(plx,u_plx)

where plx and u_plx, are the parallax and its uncertainty in arcseconds. 
The function returns the following statistics of the posterior distribution:

 	* Maximum-A-Posteriori 
 	* Mean 
 	* Standard deviation
 	* Confidence interval (95%)

Since these are not always enough, it also returns the:
 	* Integrated autocorrelation time
 	* The entire MCMC sample without the burning phase.

In practice this is all the information you might need to analyse the posterior distribution.