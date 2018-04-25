.. p2d documentation master file, created by
   sphinx-quickstart on Mon Apr 16 12:20:28 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to p2d's documentation!
===============================

This document will help you through the installation and runing of the parallax-to-distance (p2d.py) code.

The main documentation is organised in the following sections:

* :ref:`Installation`
* :ref:`Usage`
* :ref:`Priors`

Introduction
------------

As you may already know, measuring distances to the objects in the sky is one of the main goals of astronomy since its
origins. To obtain distances, astronomers usually measure the parallax. However, transforming parallaxes into distances is not always
as straight forward as inverting them (see Coryn and Xavi papers).

Formally and briefely, obtaining distances from parallaxes demands probabilistic inference. We are faced with the problem of inferring the distribution
of distance given that of the observed parallax. In the Bayesian approach, the one used by p2d code, the inference of the posterior distribution of the distance
needs the specification of :ref:`Priors` and the *Likelihood*. Therefore,


.. math::
   :nowrap:

   \begin{equation}
      p(d | \omega, \sigma_{\omega}) = \frac{p(\omega | d,\sigma_{\omega})\cdot p(d|\sigma_{\omega})}{p(\omega |\sigma_{\omega}) }
   \end{equation}


where :math:`\omega` and :math:`\sigma_{\omega}`, are the observed parallax and its uncertainty, and :math:`d` is the true distance (i.e. the one we want to infer).
In general, our prior believes do not depend on the parallax uncertainties,thus :math:`p(d|\sigma_{\omega})=p(d)`, and the denominator is a normalisation constant of no interest in the present case.


Likelihood
------------
The p2d code assumes that the observed parallax is normally distributed with median and standar deviation given by the observed value and its uncertainty, respectively.
It means that the Likelihood of the data, given the true distance, is assumed to be normal, without any systematic bias nor extra terms for underestimated uncertainties. Thus, the Likelihood is defined as,

.. math::
   :nowrap:

   \begin{equation}
      p(\omega | d,\sigma_{\omega})=\frac{1}{\sqrt{2\cdot \pi \cdot \sigma_{\omega}^2}}\cdot e^{-\frac{(\omega-1/d)^2}{2\cdot \sigma_{\omega}^2}}
   \end{equation}

Posterior of the distance
-------------------------
The prosterior distribution of the distance is sampled by the *emcee* package (Daniel Foreman-Mckey reference), which is an affine invariante MCMC sampler.
The logarithm of the posterior distribution for each distance value is computed and passed to *emcee*. This later runs for a certain number of iterations, a number specified by you. You are responsible to verify that the samples have converged to the target distribution (the posterior). To help you asses this convergence, the p2d returns the integrated autocorrelation time of the Markov Chain. In a nutshell, if this time (actually mesured in iterations) is far lower than the total number of iterations, then you can be pretty sure your sample has converged. 

Since it is well known that the posterior distribution can be biased by the prior, the function :ref:`analyse_priors` will help you to decide which prior distribution provides the less biased estimate of the distance distribution. The p2d comes with several informative and non-informative priors that you can test, according to your specific needs. 

Before rushing to obtain distance estimates, run some simulations with the :ref:`analyse_priors` code and decide the best values for:
	
	* The prior distribution and its scale parameter.
	* The number of *emcee* iterations.

Go ahead and remember Bayes' words, "with great power comes great responsability" (or it was Laplace who said it? taken from Anrew Gelman's paper "Bayes: Radical, Liberal or Conservative?")



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Installation
   Priors
   Usage
   analyse_priors

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


