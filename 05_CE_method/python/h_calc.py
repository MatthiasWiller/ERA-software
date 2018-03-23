import numpy as np
import scipy as sp
"""
---------------------------------------------------------------------------
Basic algorithm
---------------------------------------------------------------------------
Created by:
Sebastian Geyer (s.geyer@tum.de)
Matthias Willer (matthias.willer@tum.de)
Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2018-03
---------------------------------------------------------------------------
Input:
* X  :
* mu :
* Si :
* Pi :
---------------------------------------------------------------------------
Output:
* h  : 
---------------------------------------------------------------------------
"""
def h_calc(X, mu, Si, Pi):
    N     = len(X)
    k_tmp = len(Pi)
    if k_tmp == 1:
        mu = mu.squeeze()
        h = sp.stats.multivariate_normal.pdf(X,mu,Si)
    else:
        h_pre = np.zeros((N, k_tmp))
        for q in range(k_tmp):
            h_pre[:,q] = Pi[q] * sp.stats.multivariate_normal.pdf(X,mu[q,:], Si[:,:,q])
        h = np.sum(h_pre, axis=1)
    return h
## END