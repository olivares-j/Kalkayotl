from __future__ import annotations
import os
from typing import Optional, Tuple
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as tt
import scipy as sp
from pytensor.tensor import TensorVariable
import arviz as az


################################## Tails Dist ####################################
def tails_logp(value, mu, weight, std, alpha_l, alpha_r, beta_l, beta_r):
    #shape = np.shape(value)
    new_value = tt.zeros_like(value[:,0])
    #for i in range(shape[0]):
    for j in range(3):
        if j!=1:
            new_value = new_value + pm.logp(pm.Normal.dist(mu=mu[j], sigma=std[j,j]), value[:,j])
            #new_value = tt.set_subtensor(new_value[j], value_j)
        else:
            value_l = value[:,j][value[:,j] < mu[j]]
            value_r = value[:,j][value[:,j] >= mu[j]]
            #left_res = weight*pm.logp(pm.Gamma.dist(alpha=alpha_l, beta=beta_l),-(value_l - mu[j]))
            #right_res = (1-weight)*pm.logp(pm.Gamma.dist(alpha=alpha_r, beta=beta_r),value_r - mu[j])
            left_res = new_value[value[:,j] < mu[j]] + weight*pm.logp(pm.Gamma.dist(alpha=alpha_l, beta=beta_l),-(value_l - mu[j]))
            right_res = new_value[value[:,j] >= mu[j]] + (1-weight)*pm.logp(pm.Gamma.dist(alpha=alpha_r, beta=beta_r),value_r - mu[j])
            new_value = tt.set_subtensor(new_value[value[:,j] < mu[j]], left_res)
            new_value = tt.set_subtensor(new_value[value[:,j] >= mu[j]], right_res)
    return new_value

def tails_random(mu, std, weight, alpha_l, alpha_r, beta_l, beta_r, rng=None, size=None):
    size = list(size)
    res = tt.zeros(size)
    dims = size[1]
    size = size[0]
    for j in range(dims):
        if j!=1:
            res_j = rng.normal(loc=mu[j], scale=std[j][j], size=size)
            res = tt.set_subtensor(res[:,j], res_j)
        else:
            size_y_l = size
            size_y_r = size
            size_y_l = int(weight*size_y_l)
            size_y_r = int(size-size_y_l)
            left_res = mu[j] - rng.gamma(shape=alpha_l, scale=1/beta_l, size=size_y_l)
            right_res = mu[j] + rng.gamma(shape=alpha_r, scale=1/beta_r, size=size_y_r)
            res_j = tt.concatenate([left_res, right_res],axis=0)
            res = tt.set_subtensor(res[:,j], res_j)
    return res

class TailsDist():
    def __init__(self, name, mu, std, weight, alpha_l, alpha_r, beta_l, beta_r, *args, **kwargs):
        pm.CustomDist.__init__(name, mu, std, weight, alpha_l, alpha_r, beta_l, beta_r, logp=tails_logp, random=tails_random, *args, **kwargs)
    
    def dist(name, mu, std, weight, alpha_l, alpha_r, beta_l, beta_r, *args, **kwargs):
        return pm.CustomDist.dist(mu, std, weight, alpha_l, alpha_r, beta_l, beta_r, logp=tails_logp, random=tails_random, class_name=name, *args, **kwargs)
        

def np_tails_logp(value, mu, std, weight, alpha_l, alpha_r, beta_l, beta_r):
    #shape = np.shape(value)
    #for i in range(shape[0]):
    value_aux = np.zeros_like(value[:,0])
    for j in range(3):
        if j!=1:
            value_aux = value_aux + sp.stats.norm.logpdf(x=value[:,j], loc=mu[j], scale=std[j][j])
        else:
            value_l = value[:,j][value[:,j] < mu[j]].copy()
            value_r = value[:,j][value[:,j] >= mu[j]].copy()
            value_aux[value[:,j] < mu[j]] = value_aux[value[:,j] < mu[j]] + weight*sp.stats.gamma.logpdf(x=-(value_l - mu[j]), a=alpha_l, scale=1/beta_l)
            value_aux[value[:,j] >= mu[j]] = value_aux[value[:,j] >= mu[j]] + (1-weight)*sp.stats.gamma.logpdf(x=value_r - mu[j], a=alpha_r, scale=1/beta_r)
    return value_aux

def np_tails_random(mu, std, weight, alpha_l, alpha_r, beta_l, beta_r, rng=None, size=None):
    size = list(size)
    res = np.zeros(size)
    dims = size[1]
    size = size[0]
    for j in range(dims):
        if j!=1:
            res[:,j] = rng.normal(loc=mu[j], scale=std[j][j], size=size)
        else:
            size_y_l = size
            size_y_r = size
            size_y_l = int(weight*size_y_l)
            size_y_r = int(size-size_y_l)
            left_res = mu[j] - rng.gamma(shape=alpha_l, scale=1/beta_l, size=size_y_l)
            right_res = mu[j] + rng.gamma(shape=alpha_r, scale=1/beta_r, size=size_y_r)
            res[:,j] = np.concatenate((left_res, right_res), axis=0)
    return res

def test_tails_random_np(verbose:bool=False, confidence:float=0.90):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    from scipy import stats
    mu = np.random.randint(-10,10, size=3)
    std = np.eye(3)*np.random.random(size=3)*10
    weight = np.random.random()
    alpha_l = np.random.randint(5, 15)
    alpha_r = np.random.randint(5, 15)
    beta_l = np.random.random()+1e-2
    beta_r = np.random.random()+1e-2
    size = (1000,3)
    rand_extr = np_tails_random(mu=mu, std=std, weight=weight, alpha_l=alpha_l, alpha_r=alpha_r, beta_l=beta_l, beta_r=beta_r, rng=np.random.default_rng(np.random.randint(1000)), size=size)
    print("========= Testing NP Random of TailsDist ===============")
    print("------------------------- X ----------------------------")
    rand_extr_x = rand_extr[:,0]
    ks_test_x = stats.kstest(rand_extr_x, stats.norm(loc=mu[0], scale=std[0][0]).cdf)
    np.testing.assert_(ks_test_x.pvalue > (1 - confidence), msg='Fail at X Coord')
    print("                          OK                            ")
    print("--------------------------------------------------------")
    print("------------------------- Y ----------------------------")
    rand_extr_y = rand_extr[:,1]
    rand_extr_l = -(rand_extr_y[rand_extr_y < mu[1]] - mu[1])
    rand_extr_r = rand_extr_y[rand_extr_y >= mu[1]] - mu[1]
    print("-------------------- Left Gamma ------------------------")
    ks_test_l = stats.kstest(rand_extr_l, stats.gamma(a=alpha_l, scale=1/beta_l).cdf)
    np.testing.assert_(ks_test_l.pvalue > (1 - confidence), msg='Fail at Left Gamma of Y Coord')
    print("                          OK                            ")
    print("--------------------------------------------------------")
    print("-------------------- Right Gamma -----------------------")
    ks_test_r = stats.kstest(rand_extr_r, stats.gamma(a=alpha_r, scale=1/beta_r).cdf)
    np.testing.assert_(ks_test_r.pvalue > (1 - confidence), msg='Fail at Right Gamma of Y Coord')
    print("                          OK                            ")
    print("--------------------------------------------------------")
    print("------------------------- Z ----------------------------")
    rand_extr_z = rand_extr[:,2]
    ks_test_z = stats.kstest(rand_extr_z, stats.norm(loc=mu[2], scale=std[2][2]).cdf)
    np.testing.assert_(ks_test_z.pvalue > (1 - confidence), msg='Fail at Z Coord')
    print("                          OK                            ")
    print("--------------------------------------------------------")
    if verbose:
        fig, axes = plt.subplots(2,2)
        sns.histplot(rand_extr_x, kde=True, ax=axes[0,0])
        axes[0,0].set_title('X')
        sns.histplot(rand_extr_y, kde=True, ax=axes[0,1])
        axes[0,1].set_title('Y')
        sns.histplot(rand_extr_z, kde=True, ax=axes[1,0])
        axes[1,0].set_title('Z')
        ax = fig.add_subplot(2, 2, 4, projection='3d')
        ax.scatter3D(rand_extr_y, rand_extr_x, rand_extr_z)
        ax.set_xlabel('y')
        ax.set_ylabel('x')
        ax.set_zlabel('z')
        ax.set_title('Scatter')
        plt.show()

def test_tails_logp_np():
    from scipy import stats
    import matplotlib.pyplot as plt
    mu = np.random.randint(-10,10, size=3)
    std = np.eye(3)*np.random.random(size=3)*10
    weight = np.random.random()
    alpha_l = np.random.randint(5, 15)
    alpha_r = np.random.randint(5, 15)
    beta_l = np.random.random()+1e-2
    beta_r = np.random.random()+1e-2
    size = (1000,3)
    rand_extr =  np_tails_random(mu=mu, std=std, weight=weight, alpha_l=alpha_l, alpha_r=alpha_r, beta_l=beta_l, beta_r=beta_r, rng=np.random.default_rng(np.random.randint(1000)), size=size)
    log_extr = np_tails_logp(rand_extr.copy(), mu=mu, std=std, weight=weight, alpha_l=alpha_l, alpha_r=alpha_r, beta_l=beta_l, beta_r=beta_r)
    print("========= Testing NP Logp of TailsDist =================")
    rand_extr_x = rand_extr[:,0]
    rand_extr_y = rand_extr[:,1]
    rand_extr_z = rand_extr[:,2]
    rand_extr_l = rand_extr_y[rand_extr_y < mu[1]]
    rand_extr_r = rand_extr_y[rand_extr_y >= mu[1]]
    rand_extr_l = -(rand_extr_l - mu[1])
    rand_extr_r = rand_extr_r - mu[1]
    sp_log_extr = np.zeros_like(rand_extr_x)
    sp_log_extr = sp_log_extr + stats.norm.logpdf(rand_extr_x, loc=mu[0], scale=std[0][0])
    sp_log_extr[rand_extr_y < mu[1]] = sp_log_extr[rand_extr_y < mu[1]] + weight*stats.gamma.logpdf(rand_extr_l, a=alpha_l, scale=1/beta_l)
    sp_log_extr[rand_extr_y >= mu[1]] = sp_log_extr[rand_extr_y >= mu[1]] + (1-weight)*stats.gamma.logpdf(rand_extr_r, a=alpha_r, scale=1/beta_r)
    sp_log_extr = sp_log_extr + stats.norm.logpdf(rand_extr_z, loc=mu[2], scale=std[2][2])
    np.testing.assert_allclose(sp_log_extr, log_extr, rtol=1e-10, atol=0, err_msg='Fail at NP Logp')
    print("                          OK                            ")
    print("--------------------------------------------------------")

def test_tails_random(verbose:bool=False, confidence:float=0.90):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    from scipy import stats
    mu = np.random.randint(-10,10, size=3)
    std = np.eye(3)*np.random.random(size=3)
    weight = np.random.random()
    alpha_l = np.random.randint(5, 15)
    alpha_r = np.random.randint(5, 15)
    beta_l = np.random.random()+1e-2
    beta_r = np.random.random()+1e-2
    size = (1000,3)
    print("========== Testing Random of TailsDist =================")
    f = pytensor.function([], tails_random(mu=mu, std=std, weight=weight, alpha_l=alpha_l, alpha_r=alpha_r, beta_l=beta_l, beta_r=beta_r, rng=np.random.default_rng(np.random.randint(1000)), size=size))
    rand_extr = f()
    print("------------------------- X ----------------------------")
    rand_extr_x = rand_extr[:,0]
    ks_test_x = stats.kstest(rand_extr_x, stats.norm(loc=mu[0], scale=std[0][0]).cdf)
    np.testing.assert_(ks_test_x.pvalue > (1 - confidence), msg='Fail at X Coord')
    print("                          OK                            ")
    print("--------------------------------------------------------")
    print("------------------------- Y ----------------------------")
    rand_extr_y = rand_extr[:,1]
    rand_extr_l = -(rand_extr_y[rand_extr_y < mu[1]] - mu[1])
    rand_extr_r = rand_extr_y[rand_extr_y >= mu[1]] - mu[1]
    print("-------------------- Left Gamma ------------------------")
    ks_test_l = stats.kstest(rand_extr_l, stats.gamma(a=alpha_l, scale=1/beta_l).cdf)
    np.testing.assert_(ks_test_l.pvalue > (1 - confidence), msg='Fail at Left Gamma of Y Coord')
    print("                          OK                            ")
    print("--------------------------------------------------------")
    print("-------------------- Right Gamma -----------------------")
    ks_test_r = stats.kstest(rand_extr_r, stats.gamma(a=alpha_r, scale=1/beta_r).cdf)
    np.testing.assert_(ks_test_r.pvalue > (1 - confidence), msg='Fail at Right Gamma of Y Coord')
    print("                          OK                            ")
    print("--------------------------------------------------------")
    print("------------------------- Z ----------------------------")
    rand_extr_z = rand_extr[:,2]
    ks_test_z = stats.kstest(rand_extr_z, stats.norm(loc=mu[2], scale=std[2][2]).cdf)
    np.testing.assert_(ks_test_z.pvalue > (1 - confidence), msg='Fail at Z Coord')
    print("                          OK                            ")
    print("--------------------------------------------------------")
    if verbose:
        fig, axes = plt.subplots(2,2)
        sns.histplot(rand_extr[:,0], kde=True, ax=axes[0,0])
        axes[0,0].set_title('X')
        sns.histplot(rand_extr[:,1], kde=True, ax=axes[0,1])
        axes[0,1].set_title('Y')
        sns.histplot(rand_extr[:,2], kde=True, ax=axes[1,0])
        axes[1,0].set_title('Z')
        ax = fig.add_subplot(2, 2, 4, projection='3d')
        ax.scatter3D(rand_extr[:,1], rand_extr[:,0], rand_extr[:,2])
        ax.set_xlabel('y')
        ax.set_ylabel('x')
        ax.set_zlabel('z')
        ax.set_title('Scatter')
        plt.show()

def test_tails_logp():
    from scipy import stats
    import matplotlib.pyplot as plt
    mu = np.random.randint(-10,10, size=3)
    std = np.eye(3)*np.random.random(size=3)*10
    weight = np.random.random()
    alpha_l = np.random.randint(5, 15)
    alpha_r = np.random.randint(5, 15)
    beta_l = np.random.random()+1e-2
    beta_r = np.random.random()+1e-2
    size = (1000,3)
    f = pytensor.function([], tails_random(mu=mu, std=std, weight=weight, alpha_l=alpha_l, alpha_r=alpha_r, beta_l=beta_l, beta_r=beta_r, rng=np.random.default_rng(np.random.randint(1000)), size=size))
    rand_extr = f()
    fl = pytensor.function([], tails_logp(value=rand_extr, mu=mu, std=std, weight=weight, alpha_l=alpha_l, alpha_r=alpha_r, beta_l=beta_l, beta_r=beta_r))
    log_extr = fl()
    print("========== Testing Logp of TailsDist ===================")
    rand_extr_x = rand_extr[:,0]
    rand_extr_y = rand_extr[:,1]
    rand_extr_l = rand_extr_y[rand_extr_y < mu[1]]
    rand_extr_r = rand_extr_y[rand_extr_y >= mu[1]]
    rand_extr_l = -(rand_extr_l - mu[1])
    rand_extr_r = rand_extr_r - mu[1]
    rand_extr_z = rand_extr[:,2]
    sp_log_extr = np.zeros_like(rand_extr_x)
    sp_log_extr = sp_log_extr + stats.norm.logpdf(rand_extr_x, loc=mu[0], scale=std[0][0])
    sp_log_extr[rand_extr_y < mu[1]] = sp_log_extr[rand_extr_y < mu[1]] + weight*stats.gamma.logpdf(rand_extr_l, a=alpha_l, scale=1/beta_l)
    sp_log_extr[rand_extr_y >= mu[1]] = sp_log_extr[rand_extr_y >= mu[1]] + (1-weight)*stats.gamma.logpdf(rand_extr_r, a=alpha_r, scale=1/beta_r)
    sp_log_extr = sp_log_extr + stats.norm.logpdf(rand_extr_z, loc=mu[2], scale=std[2][2])
    np.testing.assert_allclose(sp_log_extr, log_extr, rtol=1e-10, atol=0, err_msg='Fail at Logp')
    print("                          OK                            ")
    print("--------------------------------------------------------")

if __name__ == "__main__":
    verbose = True
    test_tails_random_np(verbose=verbose, confidence=0.80)
    test_tails_logp_np()
    test_tails_random(verbose=verbose, confidence=0.80)
    test_tails_logp()
    # with pm.Model() as m:
    #     alpha_l = pm.HalfNormal("alpha_l", sigma=10, dtype='float64')
    #     alpha_r = pm.HalfNormal("alpha_r", sigma=10, dtype='float64')
    #     beta_l = pm.Normal("beta_l", 10, 10, dtype='float64')
    #     beta_r = pm.Normal("beta_r", 10, 10, dtype='float64')
    #     mu = pm.Normal("mu", 0, 1, dtype='float64')
    #     #cust_dist = pm.CustomDist("TailsDist", mu, alpha_l, alpha_r, beta_l, beta_r, logp=tails_logp, random=tails_random, observed=np.random.random(size=(100)))
    #     cust_dist = TailsDist("TailsDist", mu, alpha_l, alpha_r, beta_l, beta_r, observed=np.random.random(size=(10000)))

    #     #posterior = pm.sample_prior_predictive(10)
    #     posterior = pm.sample(100)
