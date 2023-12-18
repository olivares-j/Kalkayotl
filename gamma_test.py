import scipy as sp
import scipy.stats as st
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pandas as pd
import kalkayotl.Transformations as tr


def simple_test():
    print("==========  Testing Logp for 1 star  ===================")
    members = pd.read_csv('article/v2.0/ComaBer/Core/members+rvs.csv')
    #print(members)
    m_1 = members.iloc[0]
    #print(m_1)
    #print([m_1.get('ra'), m_1.get('dec'), m_1.get('parallax')])
    m_tr = tr.np_radecplx_to_galactic_xyz(np.array([[m_1.get('ra'), m_1.get('dec'), m_1.get('parallax')]]))[0]
    print(m_tr)
    log_st = st.gamma(a=2.0,scale=1./2.0).logpdf(-m_tr[1])
    print(log_st)
    f = pytensor.function([], pm.Gamma.logp(-m_tr[1],alpha=2.0, inv_beta=1./2.0))
    log_pt = f()
    print(log_pt)
    np.testing.assert_allclose(log_st, log_pt, rtol=1e-7, atol=0, err_msg='Fail at Logp')
    print("                          OK                            ")
    print("--------------------------------------------------------")

def nstars_test(n=20):
    print("========== Testing Logp for n stars  ===================")
    members = pd.read_csv('article/v2.0/ComaBer/Core/members+rvs.csv')
    for i in range(n):
        m_i = members.iloc[i]
        m_tr_i = tr.np_radecplx_to_galactic_xyz(np.array([[m_i.get('ra'), m_i.get('dec'), m_i.get('parallax')]]))[0]
        log_st_i = st.gamma(a=2.0,scale=1./2.0).logpdf(np.abs(m_tr_i[1]))
        #f = pytensor.function([], pm.Gamma.logp(np.abs(m_tr_i[1]),alpha=2.0, inv_beta=1./2.0))
        f = pytensor.function([], pm.logp(pm.Gamma.dist(alpha=2.0, beta=2.0), np.abs(m_tr_i[1])))
        
        log_pt_i = f()
        np.testing.assert_allclose(log_st_i, log_pt_i, rtol=1e-7, atol=0, err_msg=f'Fail at Logp for {i} star')
    print("                          OK                            ")
    print("--------------------------------------------------------")

def nstars_logp_test(n=20):
    print("===== Testing Different Logp for n stars  ==============")
    members = pd.read_csv('article/v2.0/ComaBer/Core/members+rvs.csv')
    for i in range(n):
        m_i = members.iloc[i]
        m_tr_i = tr.np_radecplx_to_galactic_xyz(np.array([[m_i.get('ra'), m_i.get('dec'), m_i.get('parallax')]]))[0]
        #log_st_i = st.gamma(a=2.0,scale=1./2.0).logpdf(np.abs(m_tr_i[1]))
        f = pytensor.function([], pm.Gamma.logp(np.abs(m_tr_i[1]),alpha=2.0, inv_beta=1./2.0))
        g = pytensor.function([], pm.logp(pm.Gamma.dist(alpha=2.0, beta=2.0), np.abs(m_tr_i[1])))
        
        log_st_i = f()
        log_pt_i = g()
        np.testing.assert_allclose(log_st_i, log_pt_i, rtol=1e-7, atol=0, err_msg=f'Fail at Logp for {i} star')
    print("                          OK                            ")
    print("--------------------------------------------------------")

def complete_nstars_logp_test(n=20):
    print("====== Testing Complete Logp for n stars  ==============")
    members = pd.read_csv('article/v2.0/ComaBer/Core/members+rvs.csv')
    weights = sp.stats.dirichlet.rvs(np.ones(3))[0]
    print(f'Weights: {weights}\n')
    for i in range(n):
        m_i = members.iloc[i]
        m_tr_i = tr.np_radecplx_to_galactic_xyz(np.array([[m_i.get('ra'), m_i.get('dec'), m_i.get('parallax')]]))[0]
        log_st_i = st.gamma(a=2.0,scale=1./2.0).logpdf(np.abs(m_tr_i[1]))
        
        f1 = pytensor.function([], pt.log(weights[0]) + pm.logp(pm.MvNormal.dist(mu=np.zeros(3), cov=np.eye(3)), m_tr_i))
        lp_cr = f1()

        f2 = pytensor.function([], pt.log(weights[1]) + pm.logp(pm.Gamma.dist(alpha=2.0, beta=2.0), m_tr_i[1]))
        lp_tp = f2()
        f22 = pytensor.function([], pm.MvNormal.logp(m_tr_i[::2], mu=np.zeros(2), cov=np.eye(2)))
        lp_tp += f22()

        f3 = pytensor.function([], pt.log(weights[2]) + pm.logp(pm.Gamma.dist(alpha=2.0, beta=2.0), -m_tr_i[1]))
        lp_tn = f3()
        f32 = pytensor.function([], pm.MvNormal.logp(m_tr_i[::2], mu=np.zeros(2), cov=np.eye(2)))
        lp_tn += f32()
    
        lp_cr2 = np.log(weights[0]) + st.multivariate_normal(mean=np.zeros(3),cov=np.eye(3),allow_singular=True).logpdf(m_tr_i) 

        lp_tp2 = np.log(weights[1]) + st.multivariate_normal(mean=np.zeros(2),cov=np.eye(2),allow_singular=True).logpdf(m_tr_i[::2])    
        lp_tp2 += st.gamma(a=2.0,scale=1./2).logpdf(m_tr_i[1])

        lp_tn2 = np.log(weights[2]) + st.multivariate_normal(mean=np.zeros(2),cov=np.eye(2),allow_singular=True).logpdf(m_tr_i[::2])    
        lp_tn2 += st.gamma(a=2.0,scale=1./2).logpdf(-m_tr_i[1])

        #print(f'\n\nDiff at central Logp for {i} star is: {lp_cr - lp_cr2}')
        #print(f'\n\nDiff at positive tail Logp for {i} star is: {lp_tp - lp_tp2}')
        #print(f'\n\nDiff at negative tail Logp for {i} star is: {lp_tn - lp_tn2}')
        np.testing.assert_allclose(lp_cr, lp_cr2, rtol=1e-7, atol=0, err_msg=f'Fail at central Logp for {i} star')
        np.testing.assert_allclose(lp_tp, lp_tp2, rtol=1e-7, atol=0, err_msg=f'Fail at positive tail Logp for {i} star')
        np.testing.assert_allclose(lp_tn, lp_tn2, rtol=1e-7, atol=0, err_msg=f'Fail at negative tail Logp for {i} star')
    print("                          OK                            ")
    print("--------------------------------------------------------")

def mode_test(n=20):
    print("========== Testing Mode for n stars  ===================")
    members = pd.read_csv('article/v2.0/ComaBer/Core/members+rvs.csv')
    log_lk = np.zeros((n, 3))
    for i in range(n):
        m_i = members.iloc[i]
        m_tr_i = tr.np_radecplx_to_galactic_xyz(np.array([[m_i.get('ra'), m_i.get('dec'), m_i.get('parallax')]]))[0]
        
        log_lk[i,0]  = st.multivariate_normal(mean=np.zeros(3),cov=np.eye(3),allow_singular=True).logpdf(m_tr_i)

        log_lk[i,1]  = st.multivariate_normal(mean=np.zeros(2),cov=np.eye(2),allow_singular=True).logpdf(m_tr_i[::2])
        log_lk[i,1] += st.gamma(a=2.0,scale=1./2.0).logpdf(m_tr_i[1])

        log_lk[i,2]  = st.multivariate_normal(mean=np.zeros(2),cov=np.eye(2),allow_singular=True).logpdf(m_tr_i[::2])
        log_lk[i,2] += st.gamma(a=2.0,scale=1./2.0).logpdf(-m_tr_i[1])

    idx = st.mode(log_lk.argmax(axis=1),keepdims=True)[0].flatten()
    print(idx)
    print(log_lk)
    print(log_lk.argmax(axis=1))
    print("                          OK                            ")
    print("--------------------------------------------------------")


if __name__ == "__main__":
    #simple_test()
    #nstars_test()
    #nstars_logp_test()
    #mode_test()
    complete_nstars_logp_test()