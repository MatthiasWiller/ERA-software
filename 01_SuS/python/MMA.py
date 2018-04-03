import numpy as np
import scipy as sp
from ERANataf import ERANataf
from ERADist import ERADist
"""
---------------------------------------------------------------------------
Modified Metropolis Algorithm
---------------------------------------------------------------------------
Created by:
Junyi Jiang (Junyi.Jiang@tum.de)
implemented in Python by:
Matthias Willer (matthias.willer@tum.de)
Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2018-03
---------------------------------------------------------------------------
Input:
* N       : number of samples to be generated
* u_j     : seeds used to generate the new samples 100*1 double
* H       : limit state function in the standard space
* Ns      : samples generated per chain. 10 in this case
---------------------------------------------------------------------------
Output:
* samples : new samples
* geval   : limit state function evaluations of the new samples
---------------------------------------------------------------------------
References:
1."Bayesian post-processor and other enhancements of Subset Simulation for
   estimating failure probabilities in high demensions"
   Konstantin M.Zuev et al.
   Computuers and Structures 92-93 (2012) 283-296.
based on code offered by Engineering Risk Analysis Group,TUM
---------------------------------------------------------------------------
"""
def MMA(u_j,H,b,N):
    # %% Initialize variables
    d  = np.size(u_j,axis=0)      # number of uncertain parameters(dimension),100 dimensions
    Ns = np.size(u_j,axis=1)     
    Nc = int(np.ceil(N/Ns))
    # %% initialization
    u_jp1 = list()
    g_jp1 = list()
    # %% MMA process
    P = lambda x: sp.stats.norm.pdf(x)                   # Marginal pdf of u_j
    S = lambda x: sp.stats.uniform.rvs(loc=x-1,scale=2)  # Proposal pdf
    # S = lambda x: np.random.uniform(low=x-1,high=x+1)       # Proposal pdf

    # generate a candidate state epsilon
    for i in range(Ns):  # 1000
        uu       = np.zeros((d,Nc))
        gg       = np.zeros(Nc)
        uu[:,0]  = u_j[:,i]
        gg[0]    = H(uu[:,0])
        
        for p in range(Nc-1):                               # 9 samples per chain
            xi_hat    = np.zeros(d)                         # create array for dimension-safety 
            xi_hat[:] = S(uu[:,p])                          # proposal
            r      = np.minimum(1 ,P(xi_hat)/P(uu[:,p]))
            # print(r)
            xi     = np.zeros(d)                            # candidate
            for k in range(d):                              # for each dimension  
                if np.random.rand() < r[k]:
                    xi[k]= xi_hat[k]                        # accept
                else:
                    xi[k]= uu[k,p]                          # reject

            # Evaluated the xi
            He = H(xi)               # LSF evaluation, call g_fun
            if He <= b:              # if the sample is in failure domain
                uu[:,p+1] = xi       # accept epsilon as new sample
                gg[p+1]= He
            else:                    # otherwise
                uu[:,p+1] = uu[:,p]  # reject epsilon as new sample
                gg[p+1]= gg[p]

        u_jp1.append(uu)
        g_jp1.append(gg)
    
    samples = np.zeros((d,Ns*Nc))
    geval   = np.zeros(Ns*Nc)
    
    for i in range(Ns):
        samples[:,i*Nc:(i+1)*Nc] = u_jp1[i]
        geval[i*Nc:(i+1)*Nc]     = g_jp1[i]

    samples = samples[:,:N]
    geval   = geval[:N]

    return samples, geval
# %%END