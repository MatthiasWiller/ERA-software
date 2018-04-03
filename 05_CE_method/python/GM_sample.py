import numpy as np
import scipy as sp
"""
---------------------------------------------------------------------------
Algorithm to draw samples from a Gaussian-Mixture (GM) distribution
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
* mu : [npi x d]-array of means of Gaussians in the Mixture
* Si : [d x d x npi]-array of cov-matrices of Gaussians in the Mixture
* Pi : [npi]-array of weights of Gaussians in the Mixture (sum(Pi) = 1) 
* N  : number of samples to draw from the GM distribution
---------------------------------------------------------------------------
Output:
* X  : samples from the GM distribution
---------------------------------------------------------------------------
"""
def GM_sample(mu, si, pi, N):
    if np.size(mu, axis=0) == 1:
        mu = mu.squeeze()
        si = si.squeeze()
        X = sp.stats.multivariate_normal.rvs(mean=mu,
                                             cov=si,
                                             size=N)
    else:
        # Determine number of samples from each distribution
        z = np.round(pi*N)
        
        if np.sum(z) != N:
            dif    = np.sum(z)-N
            ind    = np.argmax(z)
            z[ind] = z[ind]-dif
        
        z = z.astype(int) # integer conversion

        # Generate samples
        d   = np.size(mu, axis=1)
        X   = np.zeros([N, d])
        ind = 0
        for p in range(len(pi)):
            X[ind:ind+z[p],:] = sp.stats.multivariate_normal.rvs(mean=mu[p,:], 
                                                                 cov=si[:,:,p], 
                                                                 size=z[p]).reshape(-1,d)
            ind               = ind+z[p]
    
    return X
# %% END