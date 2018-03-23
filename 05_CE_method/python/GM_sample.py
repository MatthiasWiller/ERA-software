import numpy as np
import scipy as sp
"""
---------------------------------------------------------------------------
Basic algorithm
---------------------------------------------------------------------------
Created by:
Sebastian Geyer (s.geyer@tum.de)
implemented in Python by:
Matthias Willer (matthias.willer@tum.de)
Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2018-02
---------------------------------------------------------------------------
Input:
* mu :
* Si :
* Pi :
* N  :
---------------------------------------------------------------------------
Output:
* X  : 
---------------------------------------------------------------------------
"""
def GM_sample(mu, Si, Pi, N):
    if np.size(mu, axis=1) == 1:
        mu = mu.squeeze()
        X = sp.stats.multivariate_normal.rvs(mean=mu,
                                             cov=Si,
                                             size=N)
    else:
        # Determine number of samples from each distribution
        z = np.round(Pi*N)
        
        if np.sum(z) != N:
            dif    = np.sum(z)-N
            ind    = np.argmax(z)
            z[ind] = z[ind]-dif
        
        z = z.astype(int) # integer conversion

        # Generate samples
        X   = np.zeros([N, np.size(mu, axis=1)])
        ind = 0
        for p in range(len(Pi)):
            X[ind:ind+z[p],:] = sp.stats.multivariate_normal.rvs(mean=mu[p,:], 
                                                                 cov=Si[:,:,p], 
                                                                 size=z[p])
            ind               = ind+z[p]
    
    return X
## END