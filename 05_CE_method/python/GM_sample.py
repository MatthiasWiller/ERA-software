import numpy as np
import scipy.stats
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
    # Determine number of samples from each distribution
    z = round(Pi*N)
    
    if sum(z)!=N:
        dif    = sum(z)-N
        ind    = np.argmax(z)
        z[ind] = z[ind]-dif
    
    # Generate samples
    X   = np.zeros((N, np.size(mu, axis=1)))
    ind = 1
    for p in range(Pi):
        np                = z[p]
        X[ind:ind+np-1,:] = scipy.stats.multivariate_normal(mean=mu[p,:], cov=Si[:,:,p], size=np)
        ind               = ind+np
    
    return X
## END