import numpy as np
import scipy as sp
"""
---------------------------------------------------------------------------
Basic algorithm to calculate h for the likelihood ratio
---------------------------------------------------------------------------
Created by:
Sebastian Geyer (s.geyer@tum.de)
Implemented in Python by:
Matthias Willer (matthias.willer@tum.de)
Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2018-03
---------------------------------------------------------------------------
Input:
* X  : input samples
* mu : [npi x d]-array of means of Gaussians in the Mixture
* Si : [d x d x npi]-array of cov-matrices of Gaussians in the Mixture
* Pi : [npi]-array of weights of Gaussians in the Mixture (sum(Pi) = 1) 
---------------------------------------------------------------------------
Output:
* h  : parameters h (IS density)
---------------------------------------------------------------------------
"""
def h_calc(X, mu, si, Pi):
    N     = len(X)
    k_tmp = len(Pi)
    if k_tmp == 1:
        mu = mu.squeeze()
        si = si.squeeze()
        h = sp.stats.multivariate_normal.pdf(X,mu,si)
    else:
        h_pre = np.zeros((N, k_tmp))
        for q in range(k_tmp):
            h_pre[:,q] = Pi[q] * sp.stats.multivariate_normal.pdf(X,mu[q,:], si[:,:,q])
        h = np.sum(h_pre, axis=1)
    return h
# %% END