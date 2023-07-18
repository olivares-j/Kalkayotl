from __future__ import annotations
import os
from typing import Optional, Tuple
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as tt
import scipy as sp
from pytensor.tensor import TensorVariable
#import arviz as az


################################## Tails Dist ####################################
def tails_logp(value: TensorVariable, mu:TensorVariable,alpha_l:TensorVariable, alpha_r:TensorVariable, beta_l:TensorVariable, beta_r:TensorVariable):
    value_l = value[value < mu]
    value_r = value[value >= mu]
    left_res = pm.logp(pm.Gamma.dist(alpha=alpha_l, beta=beta_l),-(value_l - mu))
    right_res = pm.logp(pm.Gamma.dist(alpha=alpha_r, beta=beta_r),value_r - mu)
    new_value = value
    new_value = tt.set_subtensor(value[value < mu], left_res)
    new_value = tt.set_subtensor(value[value >= mu], right_res)
    return new_value

def tails_random(mu:TensorVariable, alpha_l:TensorVariable, alpha_r:TensorVariable, beta_l:TensorVariable, beta_r:TensorVariable, rng:Optional[np.random.Generator]=None, size:Optional[Tuple[int]]=None):
    if size is not None:
        size_list = list(size)
        size_list[0] = int(size_list[0]/2)
        size = tuple(size_list)
    left_res = mu - rng.gamma(shape=alpha_l, scale=1/beta_l, size=size)
    right_res = mu + rng.gamma(shape=alpha_r, scale=1/beta_r, size=size)
    return tt.concatenate([left_res, right_res],axis=0)

class TailsDist():
    def __init__(self, name, mu, alpha_l, alpha_r, beta_l, beta_r, *args, **kwargs):
        pm.CustomDist.__init__(self, name, mu, alpha_l, alpha_r, beta_l, beta_r, logp=tails_logp, random=tails_random, *args, **kwargs)
    
    def dist(name, mu, alpha_l, alpha_r, beta_l, beta_r, *args, **kwargs):
        return pm.CustomDist.dist(mu, alpha_l, alpha_r, beta_l, beta_r, logp=tails_logp, random=tails_random, class_name=name, *args, **kwargs)
        

def np_tails_logp(value, mu, alpha_l, alpha_r, beta_l, beta_r):
    value_l = value[value < mu]
    value_r = value[value >= mu]
    value[value < mu] = sp.stats.gamma.logpdf(x=-(value_l - mu), a=alpha_l, scale=1/beta_l)
    value[value >= mu] = sp.stats.gamma.logpdf(x=value_r-mu, a=alpha_r, scale=1/beta_r)
    return value

def np_tails_random(mu, alpha_l, alpha_r, beta_l, beta_r, rng=None, size=None):
    if size is not None:
        size_list = list(size)
        size_list[0] = int(size_list[0]/2)
        size = tuple(size_list)
    left_res = mu - rng.gamma(shape=alpha_l, scale=1/beta_l, size=size)
    right_res = mu + rng.gamma(shape=alpha_r, scale=1/beta_r, size=size)
    return np.concatenate((left_res, right_res))

# with pm.Model() as m:
#     alpha_l = pm.HalfNormal("alpha_l", sigma=10, dtype='float64')
#     alpha_r = pm.HalfNormal("alpha_r", sigma=10, dtype='float64')
#     beta_l = pm.Normal("beta_l", 10, 10, dtype='float64')
#     beta_r = pm.Normal("beta_r", 10, 10, dtype='float64')
#     mu = pm.Normal("mu", 0, 1, dtype='float64')
#     #cust_dist = pm.CustomDist("TailsDist", mu, alpha_l, alpha_r, beta_l, beta_r, logp=tails_logp, random=tails_random, observed=np.random.random(size=(100)))
#     cust_dist = TailsDist("TailsDist", mu, alpha_l, alpha_r, beta_l, beta_r, observed=np.random.random(size=(100)))

    #posterior = pm.sample_prior_predictive(10)
    #posterior = pm.sample(5)verbose:bool=False

    #az.plot_trace(posterior, combined=False)

def test_tails_random_np(verbose:bool=False, confidence:float=0.90):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    from scipy import stats
    mu = np.random.randint(-10,10)
    alpha_l = np.random.randint(5, 15)
    alpha_r = np.random.randint(5, 15)
    beta_l = np.random.random()+1e-2
    beta_r = np.random.random()+1e-2
    rand_extr =  np_tails_random(mu=mu, alpha_l=alpha_l, alpha_r=alpha_r, beta_l=beta_l, beta_r=beta_r, rng=np.random.default_rng(np.random.randint(1000)), size=(1000,))
    print("========= Testing Random of TailsDist NP  ===============")
    rand_extr_l = -(rand_extr[rand_extr < mu] - mu)
    rand_extr_r = rand_extr[rand_extr >= mu] - mu
    print("-------------------- Left Gamma ------------------------")
    ks_test_l = stats.kstest(rand_extr_l, stats.gamma(a=alpha_l, scale=1/beta_l).cdf)
    np.testing.assert_(ks_test_l.pvalue > (1 - confidence), msg='Fail at Left Gamma')
    print("                          OK                            ")
    print("--------------------------------------------------------")
    print("-------------------- Right Gamma -----------------------")
    ks_test_r = stats.kstest(rand_extr_r, stats.gamma(a=alpha_r, scale=1/beta_r).cdf)
    np.testing.assert_(ks_test_r.pvalue > (1 - confidence), msg='Fail at Right Gamma')
    print("                          OK                            ")
    print("--------------------------------------------------------")
    if verbose:
        fig, axes = plt.subplots(2,2)
        sns.kdeplot(rand_extr, fill=True, ax=axes[0,0])
        sns.histplot(rand_extr, ax=axes[0,1])
        sns.ecdfplot(rand_extr, ax=axes[1,0])
        sns.histplot(rand_extr, cumulative=True, ax=axes[1,1])
        plt.show()

def test_tails_logp_np():
    from scipy import stats
    import matplotlib.pyplot as plt
    mu = np.random.randint(1,10)
    alpha_l = np.random.randint(5, 15)
    alpha_r = np.random.randint(5, 15)
    beta_l = np.random.random()+1e-2
    beta_r = np.random.random()+1e-2
    rand_extr =  np_tails_random(mu=mu, alpha_l=alpha_l, alpha_r=alpha_r, beta_l=beta_l, beta_r=beta_r, rng=np.random.default_rng(np.random.randint(1000)), size=(10,))
    rand_extr_l = rand_extr[rand_extr < mu]
    rand_extr_r = rand_extr[rand_extr >= mu]
    log_extr_l = np_tails_logp(rand_extr_l.copy(), mu=mu, alpha_l=alpha_l, alpha_r=alpha_r, beta_l=beta_l, beta_r=beta_r)
    log_extr_r = np_tails_logp(rand_extr_r.copy(), mu=mu, alpha_l=alpha_l, alpha_r=alpha_r, beta_l=beta_l, beta_r=beta_r)
    print("========= Testing Logp of TailsDist  ===================")
    rand_extr_l = -(rand_extr_l - mu)
    rand_extr_r = rand_extr_r - mu
    print("-------------------- Left Gamma ------------------------")
    sp_log_extr_l = stats.gamma.logpdf(rand_extr_l, a=alpha_l, scale=1/beta_l)
    np.testing.assert_allclose(sp_log_extr_l, log_extr_l, rtol=1e-10, atol=0, err_msg='Fail at Left Gamma')
    print("                          OK                            ")
    print("--------------------------------------------------------")
    print("-------------------- Right Gamma -----------------------")
    sp_log_extr_r = stats.gamma.logpdf(rand_extr_r, a=alpha_r, scale=1/beta_r)
    np.testing.assert_allclose(sp_log_extr_r, log_extr_r, rtol=1e-10, atol=0, err_msg='Fail at Right Gamma')
    print("                          OK                            ")
    print("--------------------------------------------------------")

if __name__ == "__main__":
    test_tails_random_np(verbose=True, confidence=85)
    test_tails_logp_np()
