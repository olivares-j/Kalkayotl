from __future__ import annotations
import os
from typing import Optional, Tuple
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as tt
import scipy as sp
from pytensor.tensor import TensorVariable
from pymc.math import logsumexp
import arviz as az


################################## Tails Dist ####################################
# def tails_dist(
#     mu:TensorVariable,
#     chol:TensorVariable,
#     weight:TensorVariable,
#     alpha:TensorVariable,
#     beta:TensorVariable
#     )->TensorVariable:
#     # This must be constructed from simpler PYMC distributions


def cluster_logp(
    value:TensorVariable,   # Value where the logp will be computed
    mu:TensorVariable,      # Central cluster position
    chol_cr:TensorVariable, # Cholesky decomposition of central covariance
    chol_ta:TensorVariable, # Cholesky decomposition of tail A
    chol_tb:TensorVariable, # Cholesky decomposition of tail B
    weights:TensorVariable, # Weights of the three components
    alpha:TensorVariable    # Parameter of Gamma distribution
    ):#->TensorVariable:
        # Auxiliar Y variable
    y = value[:,1] - mu[1]

    # ---------------- Logp -----------------------------------------------------------
    lp_cr  = tt.log(weights[0]) + pm.logp(pm.MvNormal.dist(mu=mu, chol=chol_cr), value)
    lp_ta  = tt.log(weights[1]) + pm.logp(pm.Gamma.dist(alpha=alpha[0], beta=chol_ta[1,1]), -y)
    lp_tb  = tt.log(weights[2]) + pm.logp(pm.Gamma.dist(alpha=alpha[1], beta=chol_tb[1,1]),  y)
    lp_ta += pm.logp(pm.MvNormal.dist(mu=mu[::2], chol=chol_ta[::2,::2]), value[:,::2])
    lp_tb += pm.logp(pm.MvNormal.dist(mu=mu[::2], chol=chol_tb[::2,::2]), value[:,::2])
    lp     = pm.logsumexp(tt.stack([lp_cr,lp_ta,lp_tb]), axis=0)
    return lp

def cluster_random(
    mu,         # Central cluster position
    chol_cr,    # Cholesky decomposition of central covariance
    chol_ta,    # Cholesky decomposition of tail A
    chol_tb,    # Cholesky decomposition of tail B
    weights,    # Weights of the three components 
    alpha,      # Parameter of Gamma distribution
    rng=None,   # Random generator 
    size=None   # Size of the sample
    ): 
    size = list(size)
    #--------- Numbers ------------
    n_ta = int(weights[1]*size[0])
    n_tb = int(weights[2]*size[0])
    n_cr = size[0] - (n_ta+n_tb)
    #------------------------------

    #----------------- Covariances --------------------------------
    cov_cr = np.dot(chol_cr,chol_cr.T)
    cov_ta = np.dot(chol_ta[::2,::2],chol_ta[::2,::2].T)
    cov_tb = np.dot(chol_tb[::2,::2],chol_tb[::2,::2].T)
    #--------------------------------------------------------------
    
    xyz_ta = np.zeros((n_ta,size[1]))
    xyz_tb = np.zeros((n_tb,size[1]))
    xyz_cr = rng.multivariate_normal(mean=mu,cov=cov_cr,size=n_cr)
    xyz_ta[:,::2] = rng.multivariate_normal(mean=mu[::2], cov=cov_ta, size=n_ta)
    xyz_tb[:,::2] = rng.multivariate_normal(mean=mu[::2], cov=cov_tb, size=n_tb)
    xyz_ta[:,1] = mu[1] - rng.gamma(shape=alpha[0], scale=chol_ta[1,1], size=n_ta)
    xyz_tb[:,1] = mu[1] + rng.gamma(shape=alpha[1], scale=chol_tb[1,1], size=n_tb)
    
    xyz = np.concatenate((xyz_cr,xyz_ta,xyz_tb),axis=0)
    return xyz

# def tails_logp(value, mu, chol, weight, alpha,beta):
#     lp  = tt.zeros_like(value[:,0])
#     x   = value[:,1] - mu[1]
#     lp += pm.logp(pm.MvNormal.dist(mu=mu[::2], chol=chol), value[:,::2])
#     ll  = pm.logp(pm.Gamma.dist(alpha=alpha[0], beta=beta[0]), -x)
#     lr  = pm.logp(pm.Gamma.dist(alpha=alpha[1], beta=beta[1]), x)
#     lp += tt.where(value[:,1] < mu[1], ll, lr)
#     return lp

# def tails_random(mu, chol, weight, alpha, beta, rng=None, size=None):
#     size = list(size)
#     res_xz = rng.multivariate_normal(mean=mu[::2], cov=np.dot(chol,chol.T), size=size[0])
#     size_y_l = int(weight[0]*size[0])
#     size_y_r = int(size[0]-size_y_l)
#     left_res  = mu[1] - rng.gamma(shape=alpha[0], scale=1/beta[0], size=size_y_l)
#     right_res = mu[1] + rng.gamma(shape=alpha[1], scale=1/beta[1], size=size_y_r)
#     res_y = np.concatenate((left_res, right_res),axis=0)

#     res = np.zeros(size)
#     res[:,0] = res_xz[:,0]
#     res[:,1] = res_y
#     res[:,2] = res_xz[:,1]
#     return res
    # res = tt.zeros(size)
    # res = tt.set_subtensor(res[:,0], res_xz[:,0])
    # res = tt.set_subtensor(res[:,1], res_y)
    # res = tt.set_subtensor(res[:,2], res_xz[:,1])
    # return res

class TailsDist():
    def __init__(self, name, mu, chol, weight, alpha, beta, *args, **kwargs):
        pm.CustomDist.__init__(name, mu, chol, weight, alpha,  beta, logp=tails_logp, random=tails_random, *args, **kwargs)
    
    def dist(name, mu, chol, weight, alpha, beta, *args, **kwargs):
        return pm.CustomDist.dist(mu, chol, weight, alpha, beta, logp=tails_logp, random=tails_random, class_name=name, *args, **kwargs)
        

def np_tails_logp(value, mu, chol, weight, alpha_l, alpha_r, beta_l, beta_r):
    value_aux = np.zeros_like(value[:,0])
    value_aux = value_aux + sp.stats.multivariate_normal.logpdf(x=value[:,::2], mean=mu[::2], cov=np.dot(chol,chol.T))
    value_l = value[:,1][value[:,1] < mu[1]].copy()
    value_r = value[:,1][value[:,1] >= mu[1]].copy()
    value_aux[value[:,1] < mu[1]] = value_aux[value[:,1] < mu[1]] + weight*sp.stats.gamma.logpdf(x=-(value_l - mu[1]), a=alpha_l, scale=1/beta_l)
    value_aux[value[:,1] >= mu[1]] = value_aux[value[:,1] >= mu[1]] + (1-weight)*sp.stats.gamma.logpdf(x=value_r - mu[1], a=alpha_r, scale=1/beta_r)
    return value_aux

def np_tails_random(mu, chol, weight, alpha_l, alpha_r, beta_l, beta_r, rng=None, size=None):
    size = list(size)
    res = np.zeros(size)
    res_xz = rng.multivariate_normal(mean=mu[::2], cov=np.dot(chol,chol.T), size=size[0])
    res[:,0] = res_xz[:,0]
    res[:,2] = res_xz[:,1]
    size_y_l = size[0]
    size_y_r = size[0]
    size_y_l = int(weight*size_y_l)
    size_y_r = int(size[0]-size_y_l)
    left_res = mu[1] - rng.gamma(shape=alpha_l, scale=1/beta_l, size=size_y_l)
    right_res = mu[1] + rng.gamma(shape=alpha_r, scale=1/beta_r, size=size_y_r)
    res[:,1] = np.concatenate((left_res, right_res), axis=0)
    return res

def test_tails_random_np(verbose:bool=False, confidence:float=0.90):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    from scipy import stats
    import pandas as pd
    mu = np.random.randint(-10,10, size=3)
    chol = np.eye(2)*np.random.random(size=2)*10
    weight = np.random.random()
    alpha_l = np.random.randint(5, 15)
    alpha_r = np.random.randint(5, 15)
    beta_l = np.random.random()+1e-2
    beta_r = np.random.random()+1e-2
    size = (1000,3)
    rand_extr = np_tails_random(mu=mu, chol=chol, weight=weight, alpha_l=alpha_l, alpha_r=alpha_r, beta_l=beta_l, beta_r=beta_r, rng=np.random.default_rng(np.random.randint(1000)), size=size)
    print("========= Testing NP Random of TailsDist ===============")
    rand_extr_xz = rand_extr[:,::2]
    sp_rand_extr_xz = stats.multivariate_normal.rvs(mean=mu[::2], cov=np.dot(chol,chol.T), size=(size[0]))
    print("------------------------ X ---------------------------")
    ks_test_x = stats.ks_2samp(rand_extr_xz[:,0], sp_rand_extr_xz[:,0])
    np.testing.assert_(ks_test_x.pvalue > (1 - confidence), msg='Fail at X Coords')
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
    ks_test_z = stats.ks_2samp(rand_extr_xz[:,1], sp_rand_extr_xz[:,1])
    np.testing.assert_(ks_test_z.pvalue > (1 - confidence), msg='Fail at Z Coord')
    print("                          OK                            ")
    print("--------------------------------------------------------")
    if verbose:
        fig, axes = plt.subplots(2,2)
        sns.histplot(rand_extr_xz[:,0], kde=True, ax=axes[0,0])
        axes[0,0].set_title('X')
        sns.histplot(rand_extr_y, kde=True, ax=axes[0,1])
        axes[0,1].set_title('Y')
        sns.histplot(rand_extr_xz[:,1], kde=True, ax=axes[1,0])
        axes[1,0].set_title('Z')
        ax = fig.add_subplot(2, 2, 4, projection='3d')
        ax.scatter3D(rand_extr_y, rand_extr_xz[:,0], rand_extr_xz[:,1])
        ax.set_xlabel('y')
        ax.set_ylabel('x')
        ax.set_zlabel('z')
        ax.set_title('Scatter')
        plt.show()
        zeros = np.zeros_like(rand_extr_xz[:,0])
        ones = np.ones_like(sp_rand_extr_xz[:,0])
        rand_extr_xz = np.insert(rand_extr_xz, np.shape(rand_extr_xz)[1], zeros, axis=1)
        sp_rand_extr_xz = np.insert(sp_rand_extr_xz, np.shape(sp_rand_extr_xz)[1], ones, axis=1)
        all_rand_extr_xz = np.insert(rand_extr_xz, np.shape(rand_extr_xz)[0], sp_rand_extr_xz, axis=0)
        df = pd.DataFrame(all_rand_extr_xz, columns=['X', 'Z', 'source'])
        sns.jointplot(
            data=df,
            x='X',
            y='Z',
            hue='source',
            kind='kde'
        )
        plt.show()

def test_tails_logp_np():
    from scipy import stats
    import matplotlib.pyplot as plt
    mu = np.random.randint(-10,10, size=3)
    chol = np.eye(2)*np.random.random(size=2)*10
    weight = np.random.random()
    alpha_l = np.random.randint(5, 15)
    alpha_r = np.random.randint(5, 15)
    beta_l = np.random.random()+1e-2
    beta_r = np.random.random()+1e-2
    size = (1000,3)
    rand_extr =  np_tails_random(mu=mu, chol=chol, weight=weight, alpha_l=alpha_l, alpha_r=alpha_r, beta_l=beta_l, beta_r=beta_r, rng=np.random.default_rng(np.random.randint(1000)), size=size)
    log_extr = np_tails_logp(rand_extr.copy(), mu=mu, chol=chol, weight=weight, alpha_l=alpha_l, alpha_r=alpha_r, beta_l=beta_l, beta_r=beta_r)
    print("========= Testing NP Logp of TailsDist =================")
    rand_extr_xz = rand_extr[:,::2]
    rand_extr_y = rand_extr[:,1]
    rand_extr_l = rand_extr_y[rand_extr_y < mu[1]]
    rand_extr_r = rand_extr_y[rand_extr_y >= mu[1]]
    rand_extr_l = -(rand_extr_l - mu[1])
    rand_extr_r = rand_extr_r - mu[1]
    sp_log_extr = np.zeros_like(rand_extr_y)
    sp_log_extr = sp_log_extr + stats.multivariate_normal.logpdf(rand_extr_xz, mean=mu[::2], cov=np.dot(chol,chol.T))
    sp_log_extr[rand_extr_y < mu[1]] = sp_log_extr[rand_extr_y < mu[1]] + weight*stats.gamma.logpdf(rand_extr_l, a=alpha_l, scale=1/beta_l)
    sp_log_extr[rand_extr_y >= mu[1]] = sp_log_extr[rand_extr_y >= mu[1]] + (1-weight)*stats.gamma.logpdf(rand_extr_r, a=alpha_r, scale=1/beta_r)
    np.testing.assert_allclose(sp_log_extr, log_extr, rtol=1e-10, atol=0, err_msg='Fail at NP Logp')
    print("                          OK                            ")
    print("--------------------------------------------------------")

def test_tails_random(verbose:bool=False, confidence:float=0.90):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    from scipy import stats
    import pandas as pd
    mu = np.random.randint(-10,10, size=3)
    chol = np.eye(2)*np.random.random(size=2)
    weight = np.random.random()
    alpha_l = np.random.randint(5, 15)
    alpha_r = np.random.randint(5, 15)
    beta_l = np.random.random()+1e-2
    beta_r = np.random.random()+1e-2
    size = (1000,3)
    print("========== Testing Random of TailsDist =================")
    f = pytensor.function([], tails_random(mu=mu, chol=chol, weight=weight, alpha_l=alpha_l, alpha_r=alpha_r, beta_l=beta_l, beta_r=beta_r, rng=np.random.default_rng(np.random.randint(1000)), size=size))
    rand_extr = f()
    rand_extr_xz = rand_extr[:,::2]
    sp_rand_extr_xz = stats.multivariate_normal.rvs(mean=mu[::2], cov=np.dot(chol,chol.T), size=(size[0]))
    print("------------------------- X ----------------------------")
    ks_test_x = stats.ks_2samp(rand_extr_xz[:,0], sp_rand_extr_xz[:,0])
    np.testing.assert_(ks_test_x.pvalue > (1 - confidence), msg='Fail at X Coords')
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
    ks_test_z = stats.ks_2samp(rand_extr_xz[:,1], sp_rand_extr_xz[:,1])
    np.testing.assert_(ks_test_z.pvalue > (1 - confidence), msg='Fail at Z Coord')
    print("                          OK                            ")
    print("--------------------------------------------------------")
    if verbose:
        fig, axes = plt.subplots(2,2)
        sns.histplot(rand_extr_xz[:,0], kde=True, ax=axes[0,0])
        axes[0,0].set_title('X')
        sns.histplot(rand_extr_y, kde=True, ax=axes[0,1])
        axes[0,1].set_title('Y')
        sns.histplot(rand_extr_xz[:,1], kde=True, ax=axes[1,0])
        axes[1,0].set_title('Z')
        ax = fig.add_subplot(2, 2, 4, projection='3d')
        ax.scatter3D(rand_extr_y, rand_extr_xz[:,0], rand_extr_xz[:,1])
        ax.set_xlabel('y')
        ax.set_ylabel('x')
        ax.set_zlabel('z')
        ax.set_title('Scatter')
        plt.show()
        zeros = np.zeros_like(rand_extr_xz[:,0])
        ones = np.ones_like(sp_rand_extr_xz[:,0])
        rand_extr_xz = np.insert(rand_extr_xz, np.shape(rand_extr_xz)[1], zeros, axis=1)
        sp_rand_extr_xz = np.insert(sp_rand_extr_xz, np.shape(sp_rand_extr_xz)[1], ones, axis=1)
        all_rand_extr_xz = np.insert(rand_extr_xz, np.shape(rand_extr_xz)[0], sp_rand_extr_xz, axis=0)
        df = pd.DataFrame(all_rand_extr_xz, columns=['X', 'Z', 'source'])
        sns.jointplot(
            data=df,
            x='X',
            y='Z',
            hue='source',
            kind='kde'
        )
        plt.show()

def test_tails_logp():
    from scipy import stats
    import matplotlib.pyplot as plt
    mu = np.random.randint(-10,10, size=3)
    chol = np.eye(2)*np.random.random(size=2)*10
    weight = np.random.random()
    alpha_l = np.random.randint(5, 15)
    alpha_r = np.random.randint(5, 15)
    beta_l = np.random.random()+1e-2
    beta_r = np.random.random()+1e-2
    size = (1000,3)
    f = pytensor.function([], tails_random(mu=mu, chol=chol, weight=weight, alpha_l=alpha_l, alpha_r=alpha_r, beta_l=beta_l, beta_r=beta_r, rng=np.random.default_rng(np.random.randint(1000)), size=size))
    rand_extr = f()
    fl = pytensor.function([], tails_logp(value=rand_extr, mu=mu, chol=chol, weight=weight, alpha_l=alpha_l, alpha_r=alpha_r, beta_l=beta_l, beta_r=beta_r))
    log_extr = fl()
    print("========== Testing Logp of TailsDist ===================")
    rand_extr_xz = rand_extr[:,::2]
    rand_extr_y = rand_extr[:,1]
    rand_extr_l = rand_extr_y[rand_extr_y < mu[1]]
    rand_extr_r = rand_extr_y[rand_extr_y >= mu[1]]
    rand_extr_l = -(rand_extr_l - mu[1])
    rand_extr_r = rand_extr_r - mu[1]
    sp_log_extr = np.zeros_like(rand_extr_y)
    sp_log_extr = sp_log_extr + stats.multivariate_normal.logpdf(rand_extr_xz, mean=mu[::2], cov=np.dot(chol,chol.T))
    sp_log_extr[rand_extr_y < mu[1]] = sp_log_extr[rand_extr_y < mu[1]] + weight*stats.gamma.logpdf(rand_extr_l, a=alpha_l, scale=1/beta_l)
    sp_log_extr[rand_extr_y >= mu[1]] = sp_log_extr[rand_extr_y >= mu[1]] + (1-weight)*stats.gamma.logpdf(rand_extr_r, a=alpha_r, scale=1/beta_r)
    np.testing.assert_allclose(log_extr, sp_log_extr, rtol=1e-10, atol=0, err_msg='Fail at Logp')
    print("                          OK                            ")
    print("--------------------------------------------------------")

if __name__ == "__main__":
    verbose = True
    #test_tails_random_np(verbose=verbose, confidence=0.80)
    #test_tails_logp_np()
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
