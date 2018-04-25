.. _Priors:

Priors
--------------

As you may already know, in the Bayesian formalism, to estimate the posterior distribution a prior is required. These later, fromally is a probability distribution function on the parameters. However, since the posterior is noramlised by the evidence, the prior can be an improper one. Nevertheless, the prior implemented (so far) in p2d, are all of them probability distribution functions.

As you also may know, the prior is intended to probide whatever *a priori* information that you might have concerning the distribution of parameter values. Some times, this *a priori* information is negligible or even null. In the particular case of distance estimation, this *a priori* information is not null. 

We know *a priori* that there are no negative distances. This very simple and even obvious information has already reduced to halve the search space! ([0,+infinity) instad of (-infitity,+infitnity)). Another useful *a priori* information is the finite scale distance of our problem. For practical cases, our distance inference has an upper limit, that of the observable universe. In most of the practical cases of distance determination, we have a rough idea of the upper limit of our search (e.g. the solar system, the solar neighbourhood, the galaxy or the local group). This very simple lower and upper limits can be used to set a :ref:`Uniform` prior.

If we are in posession of more *a priori* information on the actual distribution of distances, we can set more complex priors. In the p2d, we provide at least three more priors which convey different degrees of information. These are given by the following distributions: :ref:`Gaussian`, :ref:`Cauchy`, and the exponentially decreasing one provided by Bailer-Jones 2015 (called :ref:`EDBJ2015` for simplicity). 


.. _Uniform:

Uniform
"""""""""

As mentioned above, this is the simplest kind of prior. On it all distances between zero and the upper limit are treated alike. All these distances have the same probability. To use this prior in p2d you only need to provide the string "Uniform" together with the scale length, wich is given in units of hectoparsecs. 

.. _Gaussian:

Gaussian
"""""""""
The Gaussian prior may help you in case your *a priori* information resambles it, like for example when  dealing with clusters, associations or star-froming regions.
For using this you need to provide the string "Gaussian" together with the mean :math:`\mu`, the standard deviation :math:`\sigma` and the scale length in units of :math:`sigma`. 

Some times, your actual *a priori* knowledge might be too restrictive, which can lead to some biases. When doing inference it is a good practice to relax your *a priori* in order to search for *knew* information. If you are interested in this topic take a look at Andrew Gelman papers on *non-informative* or *weakly informative* priors.

If the previous statement does not convince you, play with the p2d code and search for the length scale that renders the less biased statistic. 

.. _Cauchy:

Cauchy
"""""""
Inspired by Andrew Gelman papers on priors, we provide the truncated (at zero) Cauchy distribution. This distribution resembles the Gaussian one but with larger wings.
It works as a *weakly informative* prior. For using this you need to provide the string "Cauchy" together with the mean :math:`\mu`, the standard deviation :math:`\sigma` and the scale length in units of :math:`sigma`. 

.. _EDBJ2015:

EDBJ2015
"""""""""

The exponentially decreasing density law proposed by Coryn Bailer-Jones in his 2015 paper.

